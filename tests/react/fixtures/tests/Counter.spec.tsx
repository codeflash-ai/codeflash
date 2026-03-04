import { test, expect } from '@playwright/experimental-ct-react';
import {Counter} from '../Counter';
import React from 'react';

test.describe('Counter component', () => {
  test('renders with default values', async ({ mount, page }) => {
    const component = await mount(<Counter />);

    await expect(component).toContainText('Count: 0');

    const renders = await page.evaluate(() => window.__CODEFLASH_RENDERS__);
    console.log({renders})

  });

  test('renders with custom initialCount and label', async ({ mount }) => {
    const component = await mount(
      <Counter initialCount={5} label="Clicks" />
    );

    await expect(component).toContainText('Clicks: 5');
  });

  test('increments count when + is clicked', async ({ mount }) => {
    const component = await mount(<Counter initialCount={1} />);

    await component.getByText('+').click();
    await expect(component).toContainText('Count: 2');
  });

  test('decrements count when - is clicked', async ({ mount }) => {
    const component = await mount(<Counter initialCount={3} />);

    await component.getByText('-').click();
    await expect(component).toContainText('Count: 2');
  });

  test('multiple increments and decrements work correctly', async ({ mount }) => {
    const component = await mount(<Counter initialCount={0} />);

    await component.getByText('+').click();
    await component.getByText('+').click();
    await component.getByText('-').click();

    await expect(component).toContainText('Count: 1');
  });
});
