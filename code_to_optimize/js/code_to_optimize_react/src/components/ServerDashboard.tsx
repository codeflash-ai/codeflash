"use server";

/**
 * ServerDashboard - a Server Component that should be SKIPPED by the optimizer.
 *
 * Server Components (marked with "use server") run on the server and
 * don't have client-side rendering concerns like re-renders.
 * The optimizer should detect the directive and skip this file entirely.
 */

interface DashboardData {
  totalUsers: number;
  activeUsers: number;
  revenue: number;
}

export async function ServerDashboard({ orgId }: { orgId: string }) {
  const response = await fetch(`/api/dashboard/${orgId}`);
  const data: DashboardData = await response.json();

  return (
    <div>
      <h1>Dashboard</h1>
      <div>
        <p>Total Users: {data.totalUsers}</p>
        <p>Active Users: {data.activeUsers}</p>
        <p>Revenue: ${data.revenue.toFixed(2)}</p>
      </div>
    </div>
  );
}
