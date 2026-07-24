import { Suspense } from "react";
import { ResetPassword } from "@/components/account-recovery";

export default function ResetPasswordPage() {
  return <Suspense fallback={<main className="shell"><div className="panel loading">Opening reset link…</div></main>}><ResetPassword /></Suspense>;
}
