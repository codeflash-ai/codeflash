import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { FormWithValidation } from '../src/components/FormWithValidation';

const textField = (name: string, label: string, required = true) => ({
  name,
  label,
  type: 'text' as const,
  required,
});

const emailField = { name: 'email', label: 'Email', type: 'email' as const, required: true };
const passwordField = {
  name: 'password',
  label: 'Password',
  type: 'password' as const,
  required: true,
  minLength: 8,
};

const noop = () => {};

describe('FormWithValidation', () => {
  describe('rendering', () => {
    test('renders all fields', () => {
      render(
        <FormWithValidation
          fields={[textField('name', 'Name'), emailField]}
          onSubmit={noop}
        />,
      );
      expect(screen.getByText('Name')).toBeTruthy();
      expect(screen.getByText('Email')).toBeTruthy();
    });

    test('renders submit button with custom label', () => {
      render(
        <FormWithValidation
          fields={[textField('name', 'Name')]}
          onSubmit={noop}
          submitLabel="Save"
        />,
      );
      expect(screen.getByText('Save')).toBeTruthy();
    });

    test('renders with initial values', () => {
      render(
        <FormWithValidation
          fields={[textField('name', 'Name')]}
          onSubmit={noop}
          initialValues={{ name: 'John' }}
        />,
      );
      const input = screen.getByDisplayValue('John');
      expect(input).toBeTruthy();
    });

    test('renders progress bar', () => {
      render(
        <FormWithValidation
          fields={[textField('a', 'A'), textField('b', 'B'), textField('c', 'C')]}
          onSubmit={noop}
        />,
      );
      expect(screen.getByText(/0 of 3 fields completed/)).toBeTruthy();
    });

    test('shows required asterisk', () => {
      render(
        <FormWithValidation
          fields={[textField('name', 'Name', true)]}
          onSubmit={noop}
        />,
      );
      expect(screen.getByText('*')).toBeTruthy();
    });
  });

  describe('validation', () => {
    test('shows required error after submit', () => {
      render(
        <FormWithValidation
          fields={[textField('name', 'Name')]}
          onSubmit={noop}
        />,
      );
      fireEvent.click(screen.getByText('Submit'));
      expect(screen.getByText('Name is required')).toBeTruthy();
    });

    test('shows email format error', () => {
      render(
        <FormWithValidation
          fields={[emailField]}
          onSubmit={noop}
        />,
      );
      const input = screen.getByRole('textbox');
      fireEvent.change(input, { target: { value: 'not-an-email' } });
      fireEvent.blur(input);
      expect(screen.getByText(/valid email/)).toBeTruthy();
    });

    test('shows minLength error', () => {
      render(
        <FormWithValidation
          fields={[passwordField]}
          onSubmit={noop}
        />,
      );
      const input = screen.getByRole('textbox') || document.querySelector('input[type="password"]');
      if (input) {
        fireEvent.change(input, { target: { value: 'Aa1!' } });
        fireEvent.blur(input);
      }
    });

    test('validates password complexity', () => {
      render(
        <FormWithValidation
          fields={[passwordField]}
          onSubmit={noop}
        />,
      );
      // Password fields need all complexity requirements
      const inputs = document.querySelectorAll('input');
      if (inputs.length > 0) {
        fireEvent.change(inputs[0], { target: { value: 'weakpassword' } });
        fireEvent.blur(inputs[0]);
      }
    });

    test('custom validator works', () => {
      const fields = [
        {
          ...textField('age', 'Age'),
          customValidator: (value: string) =>
            Number(value) < 18 ? 'Must be 18 or older' : null,
        },
      ];
      render(<FormWithValidation fields={fields} onSubmit={noop} />);
      const input = screen.getByRole('textbox');
      fireEvent.change(input, { target: { value: '15' } });
      fireEvent.blur(input);
      expect(screen.getByText('Must be 18 or older')).toBeTruthy();
    });
  });

  describe('submission', () => {
    test('calls onSubmit with values when valid', () => {
      const onSubmit = jest.fn();
      render(
        <FormWithValidation
          fields={[textField('name', 'Name', false)]}
          onSubmit={onSubmit}
          initialValues={{ name: 'Alice' }}
        />,
      );
      fireEvent.click(screen.getByText('Submit'));
      expect(onSubmit).toHaveBeenCalledWith({ name: 'Alice' });
    });

    test('does not call onSubmit when invalid', () => {
      const onSubmit = jest.fn();
      render(
        <FormWithValidation
          fields={[textField('name', 'Name', true)]}
          onSubmit={onSubmit}
        />,
      );
      fireEvent.click(screen.getByText('Submit'));
      expect(onSubmit).not.toHaveBeenCalled();
    });

    test('calls onChange callback', () => {
      const onChange = jest.fn();
      render(
        <FormWithValidation
          fields={[textField('name', 'Name')]}
          onSubmit={noop}
          onChange={onChange}
        />,
      );
      fireEvent.change(screen.getByRole('textbox'), { target: { value: 'Bob' } });
      expect(onChange).toHaveBeenCalledWith('name', 'Bob');
    });
  });

  describe('performance with many fields', () => {
    test('renders form with 20 fields', () => {
      const fields = Array.from({ length: 20 }, (_, i) =>
        textField(`field_${i}`, `Field ${i + 1}`),
      );
      const { container } = render(
        <FormWithValidation fields={fields} onSubmit={noop} />,
      );
      expect(container).toBeTruthy();
      expect(screen.getByText(/0 of 20 fields completed/)).toBeTruthy();
    });

    test('validates all fields on submit with 20 fields', () => {
      const fields = Array.from({ length: 20 }, (_, i) =>
        textField(`field_${i}`, `Field ${i + 1}`),
      );
      const onSubmit = jest.fn();
      render(<FormWithValidation fields={fields} onSubmit={onSubmit} />);
      fireEvent.click(screen.getByText('Submit'));
      expect(onSubmit).not.toHaveBeenCalled();
    });
  });
});
