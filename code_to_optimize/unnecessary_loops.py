def process_order(order_items, menu, inventory):
    """
    Processes a restaurant order by checking availability, calculating the total cost,
    updating inventory, and printing a receipt.

    Args:
        order_items (dict): Dictionary of items ordered with quantities.
        menu (dict): Dictionary mapping items to their prices.
        inventory (dict): Dictionary mapping items to their current stock levels.

    Returns:
        str: A receipt containing the order summary and the total cost.
    """
    total_cost = 0
    receipt_lines = []
    unavailable_items = []

    # Check item availability and calculate the total cost
    for item, quantity in order_items.items():
        if item in inventory:
            # Unnecessary loop: This could be done without iterating over all menu items
            for menu_item, price in menu.items():
                if menu_item == item:
                    if inventory[item] >= quantity:
                        cost = price * quantity
                        total_cost += cost
                        inventory[item] -= quantity
                        receipt_lines.append(
                            f"{quantity}x {item} at ${price:.2f} each: ${cost:.2f}"
                        )
                    else:
                        unavailable_items.append(item)
                    break
            else:
                # Item not found in the menu
                unavailable_items.append(item)
        else:
            # Item not in inventory at all
            unavailable_items.append(item)

    # Print unavailable items
    if unavailable_items:
        receipt_lines.append("Unavailable items:")
        # Unnecessary loop: This could be replaced with a join operation
        for item in unavailable_items:
            receipt_lines.append(f"- {item}")

    # Compile the receipt
    receipt = "\n".join(receipt_lines) + f"\nTotal: ${total_cost:.2f}"

    return receipt
