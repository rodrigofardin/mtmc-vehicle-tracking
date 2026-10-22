import { TooltipProvider } from '@/components/ui/tooltip'
import { MonitoringPage } from '@/features/monitoring/MonitoringPage'

export default function App() {
  return (
    <TooltipProvider>
      <MonitoringPage />
    </TooltipProvider>
  )
}
