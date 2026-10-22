import { TooltipProvider } from '@/components/ui/tooltip'
import { MonitoringPage } from '@/features/monitoring/MonitoringPage'

export default function App() {
  return (
    <TooltipProvider delayDuration={350}>
      <MonitoringPage />
    </TooltipProvider>
  )
}
