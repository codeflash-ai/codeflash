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

  // Extract derived values once to avoid repeated property access / computations
  const { totalUsers, activeUsers, revenue } = data;
  const revenueText = "$" + revenue.toFixed(2);

  return (
    <div>
      <h1>Dashboard</h1>
      <div>
        <p>Total Users: {totalUsers}</p>
        <p>Active Users: {activeUsers}</p>
        <p>Revenue: {revenueText}</p>
      </div>
    </div>
  );
}
