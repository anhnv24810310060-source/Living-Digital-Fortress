"use client"

 import { Pie, PieChart } from "recharts"

import {
  Card,
  CardContent,
  
} from "@/components/ui/card"
import {
 type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

export const description = "A pie chart with a label"

const chartData = [
  { browser: "chrome", visitors: 275, fill: " oklch(0.81 0.10 20)" },
  { browser: "safari", visitors: 200, fill: "oklch(0.81 0.10 20)" },
  { browser: "firefox", visitors: 187, fill: "oklch(0.81 0.10 20)" },
  { browser: "edge", visitors: 173, fill: "oklch(0.81 0.10 20)" },
  { browser: "other", visitors: 90, fill: " oklch(0.81 0.10 20)" },
]

const chartConfig = {
  visitors: {
    label: "Visitors",
  },
  chrome: {
    label: "Chrome",
    color: "var(--chart-1)",
  },
  safari: {
    label: "Safari",
    color: "var(--chart-2)",
  },
  firefox: {
    label: "Firefox",
    color: "var(--chart-3)",
  },
  edge: {
    label: "Edge",
    color: "var(--chart-4)",
  },
  other: {
    label: "Other",
    color: "var(--chart-5)",
  },
} satisfies ChartConfig

export function ChartPieInteractive() {
  return (
    <Card className="flex flex-col">
      
      <CardContent className="flex-1 pb-0">
        <ChartContainer
          config={chartConfig}
          className="[&_.recharts-pie-label-text]:fill-foreground mx-auto aspect-square max-h-[250px] pb-0  "
        >
          <PieChart className="">
            <ChartTooltip     content={<ChartTooltipContent hideLabel />} />
            <Pie data={chartData}   paddingAngle={2}     dataKey="visitors" label nameKey="browser" />
          </PieChart>
        </ChartContainer> 
      </CardContent>
     
    </Card>
  )
}
