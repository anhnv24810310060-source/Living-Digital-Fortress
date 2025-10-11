"use client"

 import { CartesianGrid, Line, LineChart, XAxis } from "recharts"

import {
//   Card,
  CardContent,
 
} from "@/components/ui/card"
import {
 type  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"

export const description = "A line chart"

const chartData = [
  { month: "January", desktop: 186 },
  { month: "February", desktop: 305 },
  { month: "March", desktop: 237 },
  { month: "April", desktop: 73 },
  { month: "May", desktop: 209 },
  { month: "June", desktop: 214 },
   { month: "June", desktop: 214 },
]

const chartConfig = {
  desktop: {
    label: "Desktop",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig

export function AvgLatency() {
  return (
 
      
       
      <CardContent className="     ">
        <ChartContainer className="   " config={chartConfig}>
          <LineChart
            accessibilityLayer
            data={chartData}
            margin={{
              left: 12,
              right: 12,
            }}
           
          >
            <CartesianGrid vertical={false} />
            <XAxis
  
              dataKey="month"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              tickFormatter={(value) => value.slice(0, 3)}
            />
            <ChartTooltip
          
              cursor={false}
              content={<ChartTooltipContent hideLabel />}
            />
            <Line
            
              dataKey="desktop"
              type="natural"
              stroke="oklch(82.8% 0.111 230.318)"
              strokeWidth={2}
              dot={false}
                activeDot={{ r: 6, strokeWidth: 2, fill: "var(--background)" }}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
       
    
  )
}
