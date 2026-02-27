"use server";

interface ServerPageProps {
  id: string;
}

export async function ServerPage({ id }: ServerPageProps) {
  const data = await fetch(`/api/data/${id}`);
  const json = await data.json();

  return (
    <div>
      <h1>{json.title}</h1>
      <p>{json.description}</p>
    </div>
  );
}
