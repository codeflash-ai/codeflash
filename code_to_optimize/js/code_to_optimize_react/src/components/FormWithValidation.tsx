/**
 * FormWithValidation component - demonstrates expensive validation patterns.
 *
 * Intentionally inefficient patterns:
 * - Inline validation functions recreated every render
 * - Validation results computed without memoization
 * - Inline style objects
 * - No useCallback for handlers
 * - Inefficient regex recompilation (not cached outside render)
 *
 * Common in enterprise forms where every keystroke triggers full re-validation.
 */
import React, { useState } from 'react';

interface FormField {
  name: string;
  label: string;
  type: 'text' | 'email' | 'number' | 'password' | 'textarea';
  required: boolean;
  minLength?: number;
  maxLength?: number;
  min?: number;
  max?: number;
  pattern?: string;
  customValidator?: (value: string, allValues: Record<string, string>) => string | null;
}

interface FormWithValidationProps {
  fields: FormField[];
  onSubmit: (values: Record<string, string>) => void;
  onChange?: (field: string, value: string) => void;
  initialValues?: Record<string, string>;
  submitLabel?: string;
}

/**
 * Form component that validates all fields on every keystroke.
 * Validation is intentionally done inline without memoization.
 */
export function FormWithValidation({
  fields,
  onSubmit,
  onChange,
  initialValues = {},
  submitLabel = 'Submit',
}: FormWithValidationProps) {
  const [values, setValues] = useState<Record<string, string>>(initialValues);
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitted, setSubmitted] = useState(false);

  // Inefficient: entire validation runs on every render for all fields
  const errors: Record<string, string[]> = {};
  for (const field of fields) {
    const value = values[field.name] || '';
    const fieldErrors: string[] = [];

    if (field.required && !value.trim()) {
      fieldErrors.push(`${field.label} is required`);
    }

    if (value && field.minLength && value.length < field.minLength) {
      fieldErrors.push(`${field.label} must be at least ${field.minLength} characters`);
    }

    if (value && field.maxLength && value.length > field.maxLength) {
      fieldErrors.push(`${field.label} must be at most ${field.maxLength} characters`);
    }

    if (value && field.type === 'number') {
      const numVal = Number(value);
      if (isNaN(numVal)) {
        fieldErrors.push(`${field.label} must be a number`);
      } else {
        if (field.min !== undefined && numVal < field.min) {
          fieldErrors.push(`${field.label} must be at least ${field.min}`);
        }
        if (field.max !== undefined && numVal > field.max) {
          fieldErrors.push(`${field.label} must be at most ${field.max}`);
        }
      }
    }

    if (value && field.type === 'email') {
      // Inefficient: regex compiled on every render instead of being a constant
      const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
      if (!emailRegex.test(value)) {
        fieldErrors.push(`${field.label} must be a valid email address`);
      }
    }

    if (value && field.type === 'password') {
      // Inefficient: all these regexes compiled on every render
      if (!/[A-Z]/.test(value)) fieldErrors.push('Must contain an uppercase letter');
      if (!/[a-z]/.test(value)) fieldErrors.push('Must contain a lowercase letter');
      if (!/[0-9]/.test(value)) fieldErrors.push('Must contain a number');
      if (!/[!@#$%^&*]/.test(value)) fieldErrors.push('Must contain a special character');
    }

    if (value && field.pattern) {
      // Inefficient: regex compiled from string on every render
      const patternRegex = new RegExp(field.pattern);
      if (!patternRegex.test(value)) {
        fieldErrors.push(`${field.label} format is invalid`);
      }
    }

    if (field.customValidator) {
      const customError = field.customValidator(value, values);
      if (customError) {
        fieldErrors.push(customError);
      }
    }

    if (fieldErrors.length > 0) {
      errors[field.name] = fieldErrors;
    }
  }

  // Inefficient: computed every render
  const hasErrors = Object.keys(errors).length > 0;
  const errorCount = Object.values(errors).reduce((sum, errs) => sum + errs.length, 0);
  const completedFields = fields.filter(f => {
    const val = values[f.name] || '';
    return val.trim().length > 0 && !errors[f.name];
  }).length;
  const progress = fields.length > 0 ? (completedFields / fields.length) * 100 : 0;

  // Inefficient: function recreated every render
  const handleChange = (fieldName: string, value: string) => {
    setValues(prev => ({ ...prev, [fieldName]: value }));
    onChange?.(fieldName, value);
  };

  // Inefficient: function recreated every render
  const handleBlur = (fieldName: string) => {
    setTouched(prev => ({ ...prev, [fieldName]: true }));
  };

  // Inefficient: function recreated every render
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitted(true);
    setTouched(
      fields.reduce(
        (acc, f) => ({ ...acc, [f.name]: true }),
        {} as Record<string, boolean>,
      ),
    );
    if (!hasErrors) {
      onSubmit(values);
    }
  };

  const shouldShowError = (fieldName: string) =>
    (touched[fieldName] || submitted) && errors[fieldName];

  return (
    <form onSubmit={handleSubmit}>
      {/* Inefficient: inline style */}
      <div style={{ marginBottom: '16px' }}>
        <div
          style={{
            height: '4px',
            background: '#e0e0e0',
            borderRadius: '2px',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              height: '100%',
              width: `${progress}%`,
              background: hasErrors ? '#f44336' : '#4caf50',
              transition: 'width 0.3s',
            }}
          />
        </div>
        <small style={{ color: '#666' }}>
          {completedFields} of {fields.length} fields completed
          {errorCount > 0 && ` (${errorCount} errors)`}
        </small>
      </div>

      {fields.map(field => (
        <div key={field.name} style={{ marginBottom: '12px' }}>
          <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '4px' }}>
            {field.label}
            {field.required && <span style={{ color: 'red' }}> *</span>}
          </label>
          {field.type === 'textarea' ? (
            <textarea
              value={values[field.name] || ''}
              onChange={e => handleChange(field.name, e.target.value)}
              onBlur={() => handleBlur(field.name)}
              style={{
                width: '100%',
                padding: '8px',
                border: `1px solid ${shouldShowError(field.name) ? 'red' : '#ccc'}`,
                borderRadius: '4px',
              }}
            />
          ) : (
            <input
              type={field.type === 'password' ? 'password' : 'text'}
              value={values[field.name] || ''}
              onChange={e => handleChange(field.name, e.target.value)}
              onBlur={() => handleBlur(field.name)}
              style={{
                width: '100%',
                padding: '8px',
                border: `1px solid ${shouldShowError(field.name) ? 'red' : '#ccc'}`,
                borderRadius: '4px',
              }}
            />
          )}
          {shouldShowError(field.name) && (
            <ul style={{ color: 'red', fontSize: '12px', margin: '4px 0 0', padding: '0 0 0 16px' }}>
              {errors[field.name].map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          )}
        </div>
      ))}

      <button
        type="submit"
        disabled={submitted && hasErrors}
        style={{
          padding: '10px 24px',
          background: hasErrors && submitted ? '#ccc' : '#1976d2',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: hasErrors && submitted ? 'not-allowed' : 'pointer',
        }}
      >
        {submitLabel}
      </button>
    </form>
  );
}
