From Coq Require Import Lists.List Arith.PeanoNat Lia.
Import ListNotations.

Section QuotientKernel.

Variable H Q R : Type.
Variable response : Q -> H -> R.
Variable cost : Q -> H -> nat.
Variable mass : H -> nat.

Definition local_equiv (active : H -> Prop) (q1 q2 : Q) : Prop :=
  forall h, active h -> response q1 h = response q2 h.

Definition weak_cost_dominates (active : H -> Prop) (q1 q2 : Q) : Prop :=
  forall h, active h -> cost q1 h <= cost q2 h.

Definition strict_cost_dominates (active : H -> Prop) (q1 q2 : Q) : Prop :=
  weak_cost_dominates active q1 q2 /\
  exists h, active h /\ cost q1 h < cost q2 h.

Definition restricts (child parent : H -> Prop) : Prop :=
  forall h, child h -> parent h.

Theorem local_equiv_hereditary :
  forall parent child q1 q2,
    restricts child parent ->
    local_equiv parent q1 q2 ->
    local_equiv child q1 q2.
Proof.
  intros parent child q1 q2 Hsub Heq h Hchild.
  apply Heq.
  apply Hsub.
  exact Hchild.
Qed.

Theorem weak_cost_dominance_hereditary :
  forall parent child q1 q2,
    restricts child parent ->
    weak_cost_dominates parent q1 q2 ->
    weak_cost_dominates child q1 q2.
Proof.
  intros parent child q1 q2 Hsub Hdom h Hchild.
  apply Hdom.
  apply Hsub.
  exact Hchild.
Qed.

Theorem strict_becomes_weak_on_descendant :
  forall parent child q1 q2,
    restricts child parent ->
    strict_cost_dominates parent q1 q2 ->
    weak_cost_dominates child q1 q2.
Proof.
  intros parent child q1 q2 Hsub [Hweak _].
  eapply weak_cost_dominance_hereditary; eauto.
Qed.

Fixpoint weighted_immediate_cost (hs : list H) (q : Q) : nat :=
  match hs with
  | [] => 0
  | h :: tail => mass h * cost q h + weighted_immediate_cost tail q
  end.

Theorem weighted_immediate_cost_monotone :
  forall hs q1 q2,
    (forall h, In h hs -> cost q1 h <= cost q2 h) ->
    weighted_immediate_cost hs q1 <= weighted_immediate_cost hs q2.
Proof.
  induction hs as [|h tail IH]; intros q1 q2 Hdom; simpl.
  - lia.
  - apply Nat.add_le_mono.
    + apply Nat.mul_le_mono_l.
      apply Hdom. left. reflexivity.
    + apply IH.
      intros x Hx.
      apply Hdom. right. exact Hx.
Qed.

Definition enumerates_active (active : H -> Prop) (hs : list H) : Prop :=
  forall h, In h hs -> active h.

Theorem dominated_equivalent_root_safe_immediate :
  forall active hs q1 q2,
    enumerates_active active hs ->
    local_equiv active q1 q2 ->
    weak_cost_dominates active q1 q2 ->
    (forall h, In h hs -> response q1 h = response q2 h) /\
    weighted_immediate_cost hs q1 <= weighted_immediate_cost hs q2.
Proof.
  intros active hs q1 q2 Hactive Heq Hcost.
  split.
  - intros h Hin. apply Heq. apply Hactive. exact Hin.
  - apply weighted_immediate_cost_monotone.
    intros h Hin. apply Hcost. apply Hactive. exact Hin.
Qed.

Theorem incomparable_not_dominated_left :
  forall active q1 q2,
    (exists h, active h /\ cost q1 h < cost q2 h) ->
    (exists h, active h /\ cost q2 h < cost q1 h) ->
    ~ weak_cost_dominates active q1 q2.
Proof.
  intros active q1 q2 _ [h [Ha Hlt]] Hdom.
  specialize (Hdom h Ha).
  lia.
Qed.

Theorem incomparable_not_dominated_right :
  forall active q1 q2,
    (exists h, active h /\ cost q1 h < cost q2 h) ->
    (exists h, active h /\ cost q2 h < cost q1 h) ->
    ~ weak_cost_dominates active q2 q1.
Proof.
  intros active q1 q2 [h [Ha Hlt]] _ Hdom.
  specialize (Hdom h Ha).
  lia.
Qed.

End QuotientKernel.
