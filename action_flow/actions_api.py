from .business_logic import book_flight, cancel_ticket

def book_ticket_action(departure_city: str, arrival_city: str, travel_date: str) -> str:
    return book_flight(departure_city, arrival_city, travel_date)

def cancel_ticket_action(pnr: str) -> str:
    return cancel_ticket(pnr)
