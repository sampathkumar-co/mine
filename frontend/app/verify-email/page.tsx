import { Suspense } from "react";
import { VerifyEmail } from "@/components/account-recovery";

export default function VerifyEmailPage() {
  return <Suspense fallback={<main className="shell"><div className="panel loading">Verifying email…</div></main>}><VerifyEmail /></Suspense>;
}
