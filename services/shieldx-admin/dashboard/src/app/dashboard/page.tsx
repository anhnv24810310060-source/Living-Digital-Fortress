import { AppSidebar } from "@/components/app-sidebar"
import {
    Breadcrumb,
    BreadcrumbItem,
    BreadcrumbLink,
    BreadcrumbList,
    BreadcrumbPage,
    BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import {
    SidebarInset,
    SidebarProvider,
    SidebarTrigger,
} from "@/components/ui/sidebar"
import { ChartLineDefault } from "../components/chartTotalRequest"
import { ChartPieInteractive } from "../components/ChartPieDonutText"
import { AvgLatency } from "../components/AvgLatency"
import {
    Tooltip,
    TooltipContent,
    TooltipTrigger,
} from "@/components/ui/tooltip"
export default function Page() {
    return (
        <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
                <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-data-[collapsible=icon]/sidebar-wrapper:h-12">
                    <div className="flex items-center gap-2 px-4">
                        <SidebarTrigger className="-ml-1" />
                        <Separator
                            orientation="vertical"
                            className="mr-2 data-[orientation=vertical]:h-4"
                        />
                        <Breadcrumb>
                            <BreadcrumbList>
                                <BreadcrumbItem className="hidden md:block">
                                    <BreadcrumbLink href="#">
                                        Building Your Application
                                    </BreadcrumbLink>
                                </BreadcrumbItem>
                                <BreadcrumbSeparator className="hidden md:block" />
                                <BreadcrumbItem>
                                    <BreadcrumbPage>Data Fetching</BreadcrumbPage>
                                </BreadcrumbItem>
                            </BreadcrumbList>
                        </Breadcrumb>
                    </div>
                </header>
                <div className="w-full flex justify-start px-4 pt-4 p-5">
                    <p className="text-2xl font-sans font-bold">Overview </p>
                </div>
                <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
                    <div className="grid items-center auto-rows-min gap-4 md:grid-cols-3 p-6">
                        <div className="w-full h-auto flex p-4  flex-col  gap-2 bg-muted/50 aspect-video rounded-xl" >
                            <div className="flex    justify-center items-center   flex-row">
                                <div className="bg-blue-300/40 p-2 rounded-full">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-10 text-blue-500/90">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9.004 9.004 0 0 0 8.716-6.747M12 21a9.004 9.004 0 0 1-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 0 1 7.843 4.582M12 3a8.997 8.997 0 0 0-7.843 4.582m15.686 0A11.953 11.953 0 0 1 12 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0 1 21 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0 1 12 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 0 1 3 12c0-1.605.42-3.113 1.157-4.418" />
                                    </svg>

                                </div>
                                <div className="flex flex-col items-center justify-center w-full font-sans">
                                    <span className="text-xs uppercase tracking-wide text-gray-500">Total Requests</span>
                                    <span className="text-4xl font-bold text-gray-900">222</span>
                                </div>



                            </div>
                            <div className="  ">
                                <ChartLineDefault />

                            </div>
                        </div>
                        <div className="w-full h-auto flex p-4  flex-col  gap-2 bg-muted/50 aspect-video rounded-xl" >
                            <div className="flex    justify-center items-center   flex-row">
                                <div className="bg-red-300/40 p-2 rounded-full">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-10 text-red-500/90">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 12.75c1.148 0 2.278.08 3.383.237 1.037.146 1.866.966 1.866 2.013 0 3.728-2.35 6.75-5.25 6.75S6.75 18.728 6.75 15c0-1.046.83-1.867 1.866-2.013A24.204 24.204 0 0 1 12 12.75Zm0 0c2.883 0 5.647.508 8.207 1.44a23.91 23.91 0 0 1-1.152 6.06M12 12.75c-2.883 0-5.647.508-8.208 1.44.125 2.104.52 4.136 1.153 6.06M12 12.75a2.25 2.25 0 0 0 2.248-2.354M12 12.75a2.25 2.25 0 0 1-2.248-2.354M12 8.25c.995 0 1.971-.08 2.922-.236.403-.066.74-.358.795-.762a3.778 3.778 0 0 0-.399-2.25M12 8.25c-.995 0-1.97-.08-2.922-.236-.402-.066-.74-.358-.795-.762a3.734 3.734 0 0 1 .4-2.253M12 8.25a2.25 2.25 0 0 0-2.248 2.146M12 8.25a2.25 2.25 0 0 1 2.248 2.146M8.683 5a6.032 6.032 0 0 1-1.155-1.002c.07-.63.27-1.222.574-1.747m.581 2.749A3.75 3.75 0 0 1 15.318 5m0 0c.427-.283.815-.62 1.155-.999a4.471 4.471 0 0 0-.575-1.752M4.921 6a24.048 24.048 0 0 0-.392 3.314c1.668.546 3.416.914 5.223 1.082M19.08 6c.205 1.08.337 2.187.392 3.314a23.882 23.882 0 0 1-5.223 1.082" />
                                    </svg>


                                </div>
                                <div className="flex flex-col items-center justify-center w-full font-sans">
                                    <span className="text-xs uppercase tracking-wide text-gray-500">Threats Blocked</span>
                                    <span className="text-4xl font-bold text-gray-900"> 423</span>
                                </div>



                            </div>
                            <div className="  ">
                                <ChartPieInteractive />

                            </div>
                        </div>
                        <div className="w-full h-auto flex p-4  flex-col  gap-2 bg-muted/50 aspect-video rounded-xl" >
                            <div className="flex    justify-center items-center   flex-row">
                                <div className="bg-blue-300/40 p-2 rounded-full">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-10 text-blue-500/90">
                                        <path strokeLinecap="round" strokeLinejoin="round" d="M15.59 14.37a6 6 0 0 1-5.84 7.38v-4.8m5.84-2.58a14.98 14.98 0 0 0 6.16-12.12A14.98 14.98 0 0 0 9.631 8.41m5.96 5.96a14.926 14.926 0 0 1-5.841 2.58m-.119-8.54a6 6 0 0 0-7.381 5.84h4.8m2.581-5.84a14.927 14.927 0 0 0-2.58 5.84m2.699 2.7c-.103.021-.207.041-.311.06a15.09 15.09 0 0 1-2.448-2.448 14.9 14.9 0 0 1 .06-.312m-2.24 2.39a4.493 4.493 0 0 0-1.757 4.306 4.493 4.493 0 0 0 4.306-1.758M16.5 9a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0Z" />
                                    </svg>


                                </div>
                                <div className="flex flex-col items-center justify-center w-full font-sans">
                                    <span className="text-xs uppercase tracking-wide text-gray-500">Avg. Latency</span>
                                    <span className="text-4xl font-bold text-gray-900">222</span>
                                </div>



                            </div>
                            <div className="  ">
                                <AvgLatency />

                            </div>
                        </div>
                    </div>
                    <div className="bg-muted/50 min-h-[100vh] flex-1 rounded-xl md:min-h-min" >
                        <div className="p-8 flex flex-col ">
                            <h1 className="text-xl font-sans font-bold">System Services Health Status </h1>
                            <div className="flex flex-row gap-4 mt-4 justify-center">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>  Online</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-green-600 text-green-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Normal operation</p>
                                    </TooltipContent>
                                </Tooltip>

                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>  Warning</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-amber-600 text-amber-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Works but has issues (e.g. high latency, running out of memory).</p>
                                    </TooltipContent>
                                </Tooltip>


                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>  Offline</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-red-600 text-red-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Down, inaccessible.</p>
                                    </TooltipContent>
                                </Tooltip>



                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>  Unknown</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-gray-600 text-gray-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>








                            </div>
                        </div>
                        <div className="grid p-6  grid-cols-12 gap-4">
                            <div className="border-2 rounded-2xl   col-span-4 flex flex-col items-center justify-center gap-2 p-4">

                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Admin</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-green-600 text-green-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>

                            </div>
                            <div className="border-2 rounded-2xl  col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Auth</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-green-600 text-green-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>

                            </div>
                            <div className="border-2 rounded-2xl  col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Credits</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-gray-600 text-gray-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>
                            </div>
                            <div className="border-2 rounded-2xl col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Deception</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-gray-600 text-gray-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>
                            </div>
                            <div className="border-2 rounded-2xl col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Forensics</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-red-600 text-red-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>
                            </div>
                            <div className="border-2 rounded-2xl col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Gateway</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-gray-600 text-gray-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>
                            </div>
                            <div className="border-2 rounded-2xl col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>ML</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-green-600 text-green-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>

                            </div>
                            <div className="border-2 rounded-2xl col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Policy</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-amber-600 text-amber-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>
                            </div>
                            <div className="border-2 rounded-2xl col-span-4 flex flex-col items-center justify-center gap-2 p-4">
                                <Tooltip >
                                    <TooltipTrigger>
                                        <div className="flex justify-center items-center flex-col">
                                            <p>Sandbox</p>
                                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-3 rounded-2xl bg-gray-600 text-gray-600">
                                                <path strokeLinecap="round" strokeLinejoin="round" d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z" />
                                            </svg>
                                        </div>

                                    </TooltipTrigger>
                                    <TooltipContent>
                                        <p>Status unknown or disabled.</p>
                                    </TooltipContent>
                                </Tooltip>
                            </div>

                        </div>
                    </div>
                    <div className="bg-muted/50 min-h-[100vh] flex-1 rounded-xl md:min-h-min" />

                </div>
            </SidebarInset>
        </SidebarProvider>
    )
}
