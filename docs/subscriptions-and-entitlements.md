# Subscriptions and entitlements

Director OS keeps payment-provider state separate from its internal credit ledger.

## Source of truth

- Stripe Checkout starts a paid subscription.
- Stripe webhooks update the workspace subscription record.
- `invoice.paid` grants the configured plan credits through the existing append-only ledger.
- The invoice ID is the ledger idempotency key, so retries cannot grant credits twice.
- Browser redirects never activate a plan or grant credits.
- Cancellation or an inactive subscription returns the workspace to the starter plan without deleting projects, members, or outputs.

## Enable Stripe

1. Create recurring Stripe Prices for each paid plan.
2. Put those Price IDs into `DIRECTOR_BILLING_PLANS_JSON`.
3. Configure the secret key and webhook signing secret.
4. Enable subscriptions.

```dotenv
DIRECTOR_SUBSCRIPTIONS_ENABLED=true
DIRECTOR_BILLING_PROVIDER=stripe
DIRECTOR_STRIPE_SECRET_KEY=sk_live_replace
DIRECTOR_STRIPE_WEBHOOK_SECRET=whsec_replace
DIRECTOR_STRIPE_PORTAL_CONFIGURATION_ID=
```

The webhook endpoint is:

```text
POST /api/v1/billing/webhooks/stripe
```

Subscribe it to at least:

- `checkout.session.completed`
- `customer.subscription.created`
- `customer.subscription.updated`
- `customer.subscription.deleted`
- `customer.subscription.paused`
- `customer.subscription.resumed`
- `invoice.paid`
- `invoice.payment_failed`

The endpoint reads the raw request body and verifies the `Stripe-Signature` header before processing any event.

## Plan catalog

`DIRECTOR_BILLING_PLANS_JSON` is a JSON object keyed by stable internal plan names. Every plan defines:

- display name and description;
- Stripe Price ID for paid checkout;
- credits granted for each paid subscription invoice;
- maximum source and pickup clips;
- maximum target duration;
- maximum workspace seats;
- maximum Director tier.

The catalog must include `starter`. A blank or null `price_id` leaves a plan visible but prevents checkout, which is useful during staged rollout.

## Enforcement

Plan limits are checked at multiple boundaries:

- project creation checks target duration and Director tier;
- production start rechecks the contract and uploaded source count;
- workspace invitations count both current members and outstanding invitations;
- credit reservation still runs before queue acceptance.

Downgrades never remove existing members or projects. They block new over-limit work until usage is back within the active plan.

## Customer portal

Workspace owners can create an on-demand Stripe customer-portal session. Portal links are short-lived and are never stored as reusable credentials. Owners use the portal for payment methods, invoices, upgrades, downgrades, and cancellation according to the Stripe portal configuration.

## Webhook retention and privacy

Director OS stores one idempotency record per provider event. It stores only the event type and minimal object identifiers needed for reconciliation rather than retaining the full billing payload.

Failed events remain retryable. Processed and ignored events return success on duplicate delivery.

## Testing

The backend suite covers:

- starter duration and seat limits;
- active-plan upgrades and cancellation downgrade;
- invoice credit grants;
- duplicate webhook delivery;
- one ledger entry per paid invoice;
- migration to the subscription schema head.

Before launch, perform a Stripe test-mode rehearsal covering Checkout, portal access, renewal, failed payment, cancellation, and webhook replay.
