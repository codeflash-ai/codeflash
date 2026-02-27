import React from 'react';

interface UserCardProps {
  name: string;
  email: string;
  role: string;
  onEdit: (email: string) => void;
}

export function UserCard({ name, email, role, onEdit }: UserCardProps) {
  return (
    <div
      style={{ padding: '16px', border: '1px solid #ccc', borderRadius: '8px' }}
      className="user-card"
    >
      <h3 style={{ margin: 0 }}>{name}</h3>
      <p>{email}</p>
      <span style={{ color: role === 'admin' ? 'red' : 'blue' }}>{role}</span>
      <button onClick={() => onEdit(email)}>Edit</button>
    </div>
  );
}
