// SPDX-License-Identifier: GPL-2.0

#include <linux/generic-radix-tree.h>
#include <linux/mm.h>
#include <linux/percpu.h>
#include <linux/rcu_pending.h>
#include <linux/slab.h>
#include <linux/srcu.h>
#include <linux/vmalloc.h>

#include "rcu.h"

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
	struct rcu_head			**cursor;
	/* Overflow list, if radix tree allocation fails */
	struct rcu_head			*list;
};

struct rcu_pending_pcpu {
	struct rcu_pending		*parent;
	spinlock_t			lock;
	int				cpu:31;
	bool				rcu_armed:1;
	unsigned long			seq;
	struct rcu_pending_seq		objs[NUM_ACTIVE_RCU_POLL_OLDSTATE];
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
	/* XXX: enqueue cb to when oldest seq completes */
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
					      unsigned long flags)
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
	objs_p->cursor	= NULL;
	objs_p->list	= NULL;
	spin_unlock_irqrestore(&p->lock, flags);

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

static bool process_finished_items(struct rcu_pending *pending,
				   struct srcu_struct *srcu,
				   struct rcu_pending_pcpu *p,
				   unsigned long new_seq,
				   unsigned long flags)
{
	unsigned long old_seq = p->seq;

	int keep = max(0, 2 - (int) ((new_seq - old_seq) >> RCU_SEQ_CTR_SHIFT));

	for (struct rcu_pending_seq *objs = p->objs + ARRAY_SIZE(p->objs) - 1;
	     objs >= p->objs + keep;
	     --objs)
		if (!objs_empty(objs)) {
			__process_finished_items(pending, p, objs, flags);
			return true;
		}
	return false;
}

static void rcu_pending_work(struct work_struct *work)
{
	struct rcu_pending_pcpu *p =
		container_of(work, struct rcu_pending_pcpu, work);
	struct rcu_pending *pending = p->parent;
	unsigned long seq, flags;

	do {
		spin_lock_irqsave(&p->lock, flags);
		seq = *pending->seq & ~((ulong) RCU_SEQ_STATE_MASK);
	} while (process_finished_items(pending, pending->srcu, p, seq, flags));

	BUG_ON(!p->rcu_armed);
	p->rcu_armed = false;

	if (__rcu_pending_has_pending(p))
		rcu_pending_arm_cb(pending, p);
	spin_unlock_irqrestore(&p->lock, flags);
}

static noinline struct rcu_pending_seq *
start_new_object_list(struct srcu_struct *srcu, struct rcu_pending_pcpu *p)
{
	BUG_ON(!objs_empty(&p->objs[1]));

	/* start a new grace period */
	__start_poll_synchronize_rcu(srcu);

	/* we can't do a straight object copy because of the
	 * lock in the genradix */
	p->objs[1].objs.tree	= p->objs[0].objs.tree;
	p->objs[1].nr		= p->objs[0].nr;
	p->objs[1].cursor	= p->objs[0].cursor;
	p->objs[1].list		= p->objs[0].list;

	p->objs[0].objs.tree.root = NULL;
	p->objs[0].nr		= 0;
	p->objs[0].list		= NULL;
	p->objs[0].cursor	= NULL;
	return p->objs;
}

static __always_inline struct rcu_pending_seq *
get_object_list(struct rcu_pending *pending, struct srcu_struct *srcu,
		struct rcu_pending_pcpu **pp,
		unsigned long seq, unsigned long *flags)
{
	struct rcu_pending_pcpu *p;
relock:
	local_irq_save(*flags);
	p = *pp = this_cpu_ptr(pending->p);
	spin_lock(&p->lock);

	if (unlikely(objs_empty(p->objs) || seq > p->seq)) {
		if (process_finished_items(pending, srcu, p, seq, *flags))
			goto relock;
		start_new_object_list(srcu, p);
		p->seq = seq;
		return p->objs;
	}

	ulong idx = ((long) (p->seq - seq)) >> 2;
	if (likely(idx < ARRAY_SIZE(p->objs)))
		return p->objs + idx;

	/* seq must be expired */
	return NULL;
}

static void __rcu_pending_enqueue(struct rcu_pending *pending, struct rcu_head *head, void *ptr)
{

	struct rcu_pending_pcpu *p;
	struct rcu_pending_seq *objs;
	unsigned long seq = *pending->seq & ~((ulong) RCU_SEQ_STATE_MASK), flags;
	bool may_sleep = !ptr || !head;
relock:
	objs = get_object_list(pending, NULL, &p, seq, &flags);
	if (unlikely(!objs)) {
		spin_unlock_irqrestore(&p->lock, flags);
		if (ptr)
			kvfree(ptr);
		else
			pending->process(pending, head);
		return;
	}

	if (unlikely(!objs->cursor)) {
		objs->cursor = genradix_ptr_alloc_inlined(&objs->objs, objs->nr,
							  GFP_ATOMIC|__GFP_NOWARN);
		if (unlikely(!objs->cursor)) {
			if (may_sleep) {
				spin_unlock_irqrestore(&p->lock, flags);

				gfp_t gfp = GFP_KERNEL;
				if (!head)
					gfp |= __GFP_NOFAIL;

				may_sleep = genradix_ptr_alloc(&objs->objs, objs->nr, gfp) != NULL;
				goto relock;
			} else {
				if (ptr)
					head->func = ptr;
				head->next = objs->list;
				objs->list = head;
				goto done;
			}
		}
	}

	*objs->cursor++ = ptr;
	if (!(((ulong) objs->cursor) & (GENRADIX_NODE_SIZE - 1)))
		objs->cursor = NULL;
	objs->nr++;
done:
	rcu_pending_arm_cb(pending, p);
	spin_unlock_irqrestore(&p->lock, flags);
}

void rcu_pending_enqueue(struct rcu_pending *pending, struct rcu_head *obj)
{
	__rcu_pending_enqueue(pending, obj, NULL);
}

static struct rcu_head *rcu_pending_pcpu_dequeue(struct rcu_pending_pcpu *p)
{
	struct rcu_head *ret = NULL;

	spin_lock_irq(&p->lock);
	for (struct rcu_pending_seq *objs = p->objs + ARRAY_SIZE(p->objs) - 1;
	     objs >= p->objs;
	     --objs) {
		if (objs->nr) {
			ret = *genradix_ptr(&objs->objs, --objs->nr);
			objs->cursor = NULL;
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
	pending->seq	= rcu_get_seq();
	pending->process = process;

	return 0;
}

#ifndef CONFIG_TINY_RCU
/* kvfree_rcu */

static struct rcu_pending kvfree_rcu_pending[2];

void kvfree_call_rcu(struct rcu_head *head, void *ptr)
{
	BUG_ON(!ptr);

	__rcu_pending_enqueue(&kvfree_rcu_pending[is_vmalloc_addr_inlined(ptr)], head, ptr);
}
EXPORT_SYMBOL_GPL(kvfree_call_rcu);

void __init kvfree_rcu_pending_init(void)
{
	if (rcu_pending_init(&kvfree_rcu_pending[0], NULL, RCU_PENDING_KFREE_FN) ?:
	    rcu_pending_init(&kvfree_rcu_pending[1], NULL, RCU_PENDING_VFREE_FN))
		panic("%s failed\n", __func__);
}
#endif
