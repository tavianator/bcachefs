// SPDX-License-Identifier: GPL-2.0

#include <linux/generic-radix-tree.h>
#include <linux/mm.h>
#include <linux/percpu.h>
#include <linux/rcu_pending.h>
#include <linux/slab.h>
#include <linux/srcu.h>
#include <linux/vmalloc.h>

enum rcu_pending_special {
	RCU_PENDING_KFREE	= 1,
	RCU_PENDING_VFREE	= 2,
};

#define RCU_PENDING_KFREE_FN		((rcu_pending_process_fn) (ulong) RCU_PENDING_KFREE)
#define RCU_PENDING_VFREE_FN		((rcu_pending_process_fn) (ulong) RCU_PENDING_VFREE)

static inline unsigned long __get_state_synchronize_rcu(struct srcu_struct *ssp)
{
	return ssp
		? get_state_synchronize_srcu(ssp)
		: get_state_synchronize_rcu();
}

static inline unsigned long __start_poll_synchronize_rcu(struct srcu_struct *ssp)
{
	return ssp
		? start_poll_synchronize_srcu(ssp)
		: start_poll_synchronize_rcu();
}

static inline bool __poll_state_synchronize_rcu(struct srcu_struct *ssp, unsigned long cookie)
{
	return ssp
		? poll_state_synchronize_srcu(ssp, cookie)
		: poll_state_synchronize_rcu(cookie);
}

static inline void __rcu_barrier(struct srcu_struct *ssp)
{
	return ssp
		? srcu_barrier(ssp)
		: rcu_barrier();
}

static inline void __call_rcu(struct srcu_struct *ssp, struct rcu_head *rhp,
			      rcu_callback_t func)
{
	if (ssp)
		call_srcu(ssp, rhp, func);
	else
		call_rcu(rhp, func);
}

struct rcu_pending_seq {
	/*
	 * We're using a radix tree like a vector - we're just pushing elements
	 * onto the end; we're using a radix tree instead of an actual vector to
	 * avoid reallocation overhead
	 */
	GENRADIX(struct rcu_head *)	objs;
	size_t				nr;
	/* Overflow list, if radix tree allocation fails */
	struct rcu_head			*list;
	unsigned long			seq;
};

struct rcu_pending_pcpu {
	struct rcu_pending		*parent;
	spinlock_t			lock;
	int				cpu:31;
	bool				rcu_armed:1;
	struct rcu_pending_seq		objs[2];
	struct rcu_head			rcu;
	struct work_struct		work;
};

static bool objs_empty(struct rcu_pending_seq *objs)
{
	return !objs->nr && !objs->list;
}

#define for_each_object_list(_p, _objs)					\
	for (struct rcu_pending_seq *_objs = (_p)->objs;		\
	     _objs < (_p)->objs + ARRAY_SIZE(p->objs); _objs++)

static bool __rcu_pending_has_pending(struct rcu_pending_pcpu *p)
{
	for_each_object_list(p, objs)
		if (!objs_empty(objs))
			return true;
	return false;
}

static void rcu_pending_rcu_cb(struct rcu_head *rcu)
{
	struct rcu_pending_pcpu *p =
		container_of(rcu, struct rcu_pending_pcpu, rcu);

	schedule_work_on(p->cpu, &p->work);
}

static noinline void __rcu_pending_arm_cb(struct rcu_pending *pending, struct rcu_pending_pcpu *p)
{
	p->rcu_armed = true;
	__call_rcu(pending->srcu, &p->rcu, rcu_pending_rcu_cb);
}

static inline void rcu_pending_arm_cb(struct rcu_pending *pending, struct rcu_pending_pcpu *p)
{
	if (unlikely(!p->rcu_armed))
		__rcu_pending_arm_cb(pending, p);
}

static noinline void __process_finished_items(struct rcu_pending *pending,
					      struct rcu_pending_pcpu *p,
					      struct rcu_pending_seq *objs_p,
					      unsigned long flags,
					      bool locked)
{
	write_lock(&objs_p->objs.tree.free_lock);
	struct rcu_pending_seq objs = (struct rcu_pending_seq) {
		/*
		 * the genradix can only be modified with atomic instructions,
		 * since we allocate new nodes without
		 * rcu_pending_pcpu.lock
		 */
		.objs.tree.root	= xchg(&objs_p->objs.tree.root, NULL),
		.nr		= objs_p->nr,
		.list		= objs_p->list,
	};
	write_unlock(&objs_p->objs.tree.free_lock);
	objs_p->nr	= 0;
	objs_p->list	= NULL;
	if (locked)
		spin_unlock(&p->lock);
	local_irq_restore(flags);

	switch ((ulong) pending->process) {
	case RCU_PENDING_KFREE:
		for (size_t i = 0; i < objs.nr; ) {
			size_t nr_this_node = min(GENRADIX_NODE_SIZE / sizeof(void *), objs.nr - i);

			kfree_bulk(nr_this_node, (void **) genradix_ptr(&objs.objs, i));
			i += nr_this_node;
		}
		genradix_free_unlocked(&objs.objs);

		while (objs.list) {
			struct rcu_head *obj = objs.list;
			objs.list = obj->next;
			kfree(obj->func);
		}

		break;
	case RCU_PENDING_VFREE:
		for (size_t i = 0; i < objs.nr; i++)
			vfree(*genradix_ptr(&objs.objs, i));
		genradix_free_unlocked(&objs.objs);

		while (objs.list) {
			struct rcu_head *obj = objs.list;
			objs.list = obj->next;
			vfree(obj->func);
		}

		break;
	default:
		for (size_t i = 0; i < objs.nr; i++)
			pending->process(pending, *genradix_ptr(&objs.objs, i));
		genradix_free_unlocked(&objs.objs);

		while (objs.list) {
			struct rcu_head *obj = objs.list;
			objs.list = obj->next;
			pending->process(pending, obj);
		}
		break;
	}
}

static inline struct rcu_pending_seq *
get_finished_items(struct srcu_struct *srcu,
		   struct rcu_pending_pcpu *p)
{
	for_each_object_list(p, objs)
		if (!objs_empty(objs) &&
		    __poll_state_synchronize_rcu(srcu, objs->seq))
			return objs;
	return NULL;
}

static inline bool process_finished_items(struct rcu_pending *pending,
					  struct srcu_struct *srcu,
					  struct rcu_pending_pcpu *p,
					  unsigned long flags,
					  bool locked)
{
	struct rcu_pending_seq *finished = get_finished_items(srcu, p);
	if (unlikely(finished))
		__process_finished_items(pending, p, finished, flags, locked);
	return finished != NULL;
}

static void rcu_pending_work(struct work_struct *work)
{
	struct rcu_pending_pcpu *p =
		container_of(work, struct rcu_pending_pcpu, work);
	struct rcu_pending *pending = p->parent;
	unsigned long flags;

	do {
		spin_lock_irqsave(&p->lock, flags);
	} while (process_finished_items(pending, pending->srcu, p, flags, true));

	BUG_ON(!p->rcu_armed);
	p->rcu_armed = false;

	if (__rcu_pending_has_pending(p))
		rcu_pending_arm_cb(pending, p);
	spin_unlock_irqrestore(&p->lock, flags);
}

static __always_inline struct rcu_pending_seq *
get_object_list(struct rcu_pending *pending, struct srcu_struct *srcu,
		struct rcu_pending_pcpu **pp, unsigned long *flags,
		bool locked)
{
	struct rcu_pending_pcpu *p;

relock:
	local_irq_save(*flags);
	p = *pp = this_cpu_ptr(pending->p);
	if (locked)
		spin_lock(&p->lock);
process_finished:
	if (process_finished_items(pending, srcu, p, *flags, locked))
		goto relock;

	unsigned long seq = __get_state_synchronize_rcu(srcu);
	for_each_object_list(p, objs)
		if (!objs_empty(objs) && objs->seq == seq)
			return objs;

	seq = __start_poll_synchronize_rcu(srcu);
	for_each_object_list(p, objs)
		if (objs_empty(objs)) {
			objs->seq = seq;
			return objs;
		}

	/*
	 * start_poll_synchronize_rcu() raced with our previous
	 * process_finished_items(), and we now have another set of objects that
	 * are ready to be processed
	 */
	goto process_finished;
}

void rcu_pending_enqueue(struct rcu_pending *pending, struct rcu_head *obj)
{
	struct rcu_pending_pcpu *p;
	struct rcu_pending_seq *objs;
	struct rcu_head **entry;
	unsigned long flags;
	bool may_sleep = true;
relock:
	objs = get_object_list(pending, pending->srcu, &p, &flags, true);
	entry = genradix_ptr_alloc_inlined(&objs->objs, objs->nr, GFP_ATOMIC|__GFP_NOWARN);
	if (likely(entry)) {
		*entry = obj;
		objs->nr++;
	} else if (may_sleep) {
		spin_unlock_irqrestore(&p->lock, flags);
		if (!genradix_ptr_alloc(&objs->objs, objs->nr, GFP_KERNEL))
			may_sleep = false;
		goto relock;
	} else {
		obj->next = objs->list;
		objs->list = obj;
	}

	rcu_pending_arm_cb(pending, p);
	spin_unlock_irqrestore(&p->lock, flags);
}

static struct rcu_head *rcu_pending_pcpu_dequeue(struct rcu_pending_pcpu *p)
{
	struct rcu_head *ret = NULL;

	spin_lock_irq(&p->lock);
	unsigned idx = p->objs[1].seq > p->objs[0].seq;

	for (unsigned i = 0; i < 2; i++, idx ^= 1) {
		struct rcu_pending_seq *objs = p->objs + idx;

		if (objs->nr) {
			ret = *genradix_ptr(&objs->objs, --objs->nr);
			break;
		}

		if (objs->list) {
			ret = objs->list;
			objs->list = ret->next;
			break;
		}
	}
	spin_unlock_irq(&p->lock);

	return ret;
}

struct rcu_head *rcu_pending_dequeue(struct rcu_pending *pending)
{
	return rcu_pending_pcpu_dequeue(raw_cpu_ptr(pending->p));
}

struct rcu_head *rcu_pending_dequeue_from_all(struct rcu_pending *pending)
{
	struct rcu_head *ret = NULL;
	int cpu;
	for_each_possible_cpu(cpu) {
		ret = rcu_pending_pcpu_dequeue(per_cpu_ptr(pending->p, cpu));
		if (ret)
			break;
	}
	return ret;
}

static bool rcu_pending_has_pending_or_armed(struct rcu_pending *pending)
{
	int cpu;
	for_each_possible_cpu(cpu) {
		struct rcu_pending_pcpu *p = per_cpu_ptr(pending->p, cpu);
		spin_lock_irq(&p->lock);
		if (__rcu_pending_has_pending(p) || p->rcu_armed) {
			spin_unlock_irq(&p->lock);
			return true;
		}
		spin_unlock_irq(&p->lock);
	}

	return false;
}

void rcu_pending_exit(struct rcu_pending *pending)
{
	int cpu;

	if (!pending->p)
		return;

	while (rcu_pending_has_pending_or_armed(pending)) {
		__rcu_barrier(pending->srcu);

		for_each_possible_cpu(cpu) {
			struct rcu_pending_pcpu *p = per_cpu_ptr(pending->p, cpu);
			flush_work(&p->work);
		}
	}

	for_each_possible_cpu(cpu) {
		struct rcu_pending_pcpu *p = per_cpu_ptr(pending->p, cpu);

		WARN_ON(p->objs[0].nr);
		WARN_ON(p->objs[1].nr);
		WARN_ON(p->objs[0].list);
		WARN_ON(p->objs[1].list);

		genradix_free(&p->objs[0].objs);
		genradix_free(&p->objs[1].objs);
	}
	free_percpu(pending->p);
}

int rcu_pending_init(struct rcu_pending *pending,
		     struct srcu_struct *srcu,
		     rcu_pending_process_fn process)
{
	pending->p = alloc_percpu(struct rcu_pending_pcpu);
	if (!pending->p)
		return -ENOMEM;

	int cpu;
	for_each_possible_cpu(cpu) {
		struct rcu_pending_pcpu *p = per_cpu_ptr(pending->p, cpu);
		p->parent	= pending;
		p->cpu		= cpu;
		spin_lock_init(&p->lock);
		for_each_object_list(p, objs)
			genradix_init(&objs->objs);
		INIT_WORK(&p->work, rcu_pending_work);
	}

	pending->srcu = srcu;
	pending->process = process;

	return 0;
}

#ifndef CONFIG_TINY_RCU
/* kvfree_rcu */

static struct rcu_pending kfree_rcu_pending;
static struct rcu_pending vfree_rcu_pending;

void kvfree_call_rcu(struct rcu_head *head, void *ptr)
{
	struct rcu_pending *pending = is_vmalloc_addr_inlined(ptr)
		? &vfree_rcu_pending
		: &kfree_rcu_pending;

	struct rcu_pending_pcpu *p;
	struct rcu_pending_seq *objs;
	struct rcu_head **entry;
	unsigned long flags;
reget:
	objs = get_object_list(pending, NULL, &p, &flags, false);
	entry = genradix_ptr_alloc_inlined(&objs->objs, objs->nr, GFP_ATOMIC|__GFP_NOWARN);
	if (likely(entry)) {
		*entry = ptr;
		objs->nr++;
	} else if (head) {
		head->func = ptr;
		head->next = objs->list;
		objs->list = head;
	} else {
		local_irq_restore(flags);
		genradix_ptr_alloc(&objs->objs, objs->nr, GFP_KERNEL|__GFP_NOFAIL);
		goto reget;
	}

	rcu_pending_arm_cb(pending, p);
	local_irq_restore(flags);
}
EXPORT_SYMBOL_GPL(kvfree_call_rcu);

void __init kvfree_rcu_pending_init(void)
{
	if (rcu_pending_init(&kfree_rcu_pending, NULL, RCU_PENDING_KFREE_FN) ?:
	    rcu_pending_init(&vfree_rcu_pending, NULL, RCU_PENDING_VFREE_FN))
		panic("%s failed\n", __func__);
}
#endif
