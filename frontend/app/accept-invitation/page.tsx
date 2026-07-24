import { Suspense } from "react";
import { AcceptInvitation } from "@/components/account-recovery";

export default function AcceptInvitationPage() {
  return <Suspense fallback={<main className="shell"><div className="panel loading">Opening invitation…</div></main>}><AcceptInvitation /></Suspense>;
}
