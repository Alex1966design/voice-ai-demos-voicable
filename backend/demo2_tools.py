from typing import List


def calculate_order_total(items: List[float], vat_rate: float = 0.2) -> dict:
    """
    Пример функции, которую позже подвесим как tool.
    """
    subtotal = sum(items)
    vat = subtotal * vat_rate
    total = subtotal + vat
    return {
        "items": items,
        "subtotal": round(subtotal, 2),
        "vat": round(vat, 2),
        "total": round(total, 2),
        "vat_rate": vat_rate,
    }


def fake_get_weather(city: str) -> dict:
    """
    Фейковый weather tool для DEMO 2.
    """
    return {
        "city": city,
        "temperature_c": 24,
        "condition": "sunny",
    }
