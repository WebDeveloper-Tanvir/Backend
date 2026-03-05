import { NextRequest, NextResponse } from "next/server"

// Your live Razorpay keys
const KEY_ID = process.env.RAZORPAY_KEY_ID || "rzp_live_S9yC5XI9qt6E9a"
const KEY_SECRET = process.env.RAZORPAY_KEY_SECRET || "E82uvA3XxzAOr0rKQYrt97pR"

const DEVMODE_PLANS = {
  basic: {
    name: "Errorless Dev Mode Basic",
    amount: 20900, // ₹209 in paise (~$2.5)
    currency: "INR",
    description: "Code Generation + Analysis. 24-hour memory.",
  },
  premium: {
    name: "Errorless Dev Mode Premium",
    amount: 41900, // ₹419 in paise (~$5)
    currency: "INR",
    description: "Generate + Analyze + Optimize. Permanent memory.",
  },
}

export async function POST(req: NextRequest) {
  try {
    const { plan } = await req.json()
    const planData = DEVMODE_PLANS[plan as keyof typeof DEVMODE_PLANS]

    if (!planData) {
      return NextResponse.json({ error: `Invalid plan: "${plan}". Must be "basic" or "premium".` }, { status: 400 })
    }

    const auth = "Basic " + Buffer.from(`${KEY_ID}:${KEY_SECRET}`).toString("base64")

    const response = await fetch("https://api.razorpay.com/v1/orders", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: auth },
      body: JSON.stringify({
        amount: planData.amount,
        currency: planData.currency,
        receipt: `devmode_${plan}_${Date.now()}`,
        notes: { plan, platform: "errorless", feature: "devmode" },
      }),
    })

    if (!response.ok) {
      const err = await response.json()
      throw new Error(err.error?.description || `Razorpay error ${response.status}`)
    }

    const order = await response.json()

    return NextResponse.json({
      orderId: order.id,
      amount: order.amount,
      currency: order.currency,
      planName: planData.name,
      description: planData.description,
    })
  } catch (err) {
    console.error("[DevMode] create-order error:", err)
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Order creation failed" },
      { status: 500 }
    )
  }
}