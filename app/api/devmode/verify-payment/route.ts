import { NextRequest, NextResponse } from "next/server"
import crypto from "crypto"

const KEY_SECRET = process.env.RAZORPAY_KEY_SECRET || "E82uvA3XxzAOr0rKQYrt97pR"

export async function POST(req: NextRequest) {
  try {
    const { razorpay_order_id, razorpay_payment_id, razorpay_signature, plan } = await req.json()

    if (!razorpay_order_id || !razorpay_payment_id || !razorpay_signature) {
      return NextResponse.json({ error: "Missing payment fields" }, { status: 400 })
    }

    // Cryptographic verification — NEVER skip this
    const body = `${razorpay_order_id}|${razorpay_payment_id}`
    const expected = crypto
      .createHmac("sha256", KEY_SECRET)
      .update(body)
      .digest("hex")

    if (expected !== razorpay_signature) {
      console.error("[DevMode] Signature mismatch — possible fraud")
      return NextResponse.json({ error: "Invalid payment signature" }, { status: 400 })
    }

    // Optionally persist to Supabase if you have it configured
    try {
      const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
      const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY
      if (supabaseUrl && supabaseKey) {
        await fetch(`${supabaseUrl}/rest/v1/devmode_subscriptions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            apikey: supabaseKey,
            Authorization: `Bearer ${supabaseKey}`,
            Prefer: "return=minimal",
          },
          body: JSON.stringify({
            plan,
            payment_id: razorpay_payment_id,
            order_id: razorpay_order_id,
            activated_at: new Date().toISOString(),
          }),
        })
      }
    } catch (dbErr) {
      // Non-fatal — payment verified, DB log failed
      console.warn("[DevMode] DB log skipped:", dbErr)
    }

    return NextResponse.json({
      success: true,
      plan,
      paymentId: razorpay_payment_id,
      message: `Dev Mode ${plan} activated successfully!`,
    })
  } catch (err) {
    console.error("[DevMode] verify-payment error:", err)
    return NextResponse.json({ error: "Verification failed" }, { status: 500 })
  }
}