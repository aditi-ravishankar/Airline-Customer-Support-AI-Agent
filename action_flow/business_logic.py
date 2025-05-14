from datetime import datetime
from .database_utils import connect_to_mysql

def generate_new_pnr(cnx):
    cursor = cnx.cursor()
    try:
        # Fetch the last PNR (most recent booking)
        cursor.execute("SELECT PNR FROM flight_booking_data ORDER BY PNR DESC LIMIT 1")

        result = cursor.fetchone()
        
        if result:
            # Extract the numeric part of the last PNR
            last_pnr = result[0]
            last_number = int(last_pnr.replace("BIN", ""))  # Assuming PNR is always in "BIN" + number format
            new_pnr = f"BIN{last_number + 1:04d}"  # Increment the number and format it as BINXXXX
        else:
            # If no PNR exists (first booking), start with BIN0001
            new_pnr = "BIN0001"
    except Exception as e:
        print(f"Error generating PNR: {e}")
        new_pnr = None  # Or return a default/error code if needed
    finally:
        cursor.close()
        
    return new_pnr


def get_user_name(cnx, user_id):
    cursor = cnx.cursor()
    cursor.execute("SELECT Name FROM user_data WHERE ID = %s", (user_id,))
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else None

def check_flight_exists(cnx, departure_city, arrival_city):
    cursor = cnx.cursor()
    query = """
        SELECT Flight_Number FROM flight_data
        WHERE LOWER(Departure_City) = %s AND LOWER(Arrival_City) = LOWER(%s)
    """
    cursor.execute(query, (departure_city, arrival_city))
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else None

def book_flight(departure_city, arrival_city, travel_date_str):
    cnx = connect_to_mysql()
    if not cnx:
        return "DB connection failed"

    try:
        travel_date = travel_date_str
        today = datetime.today().date()
        flight_number = check_flight_exists(cnx, departure_city, arrival_city)
        user_id = 111

        if not flight_number:
            return f"❌ I'm sorry, but there no flights from {departure_city} to {arrival_city}."
        
        # Fetch User Name
        user_name = get_user_name(cnx, user_id)
        if not user_name:
            return f"❌ I'm sorry, but User with ID {user_id} is not found."

        pnr = generate_new_pnr(cnx)
        insert_query = """
            INSERT INTO flight_booking_data (User_ID, Flight_ID, Booking_Date, Travel_Date, PNR, Status)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor = cnx.cursor()
        cursor.execute(insert_query, (user_id, flight_number, today, travel_date, pnr, "Active"))
        cnx.commit()
        cursor.close()
        return (
            f"✅ Flight booked successfully for {user_name}!\n"
            f"🛫 Departure: {departure_city}\n"
            f"🛬 Arrival: {arrival_city}\n"
            f"🗓️ Travel Date: {travel_date}\n"
            f"📅 Booking Date: {today}\n"
            f"✈️ Flight Number: {flight_number}\n"
            f"🔐 PNR: {pnr}\n"
            f"📌 Status: Active\n\n"
            f"💳 A confirmation email with the payment link will be sent shortly.\n "
            f"Please complete the payment within 12 hours to secure your booking.\n "
            f"If payment is not received in time, the reservation will be automatically cancelled."
        )

    except ValueError:
        return "❌ Invalid date format. Please use DD-MM-YYYY."

    except Exception as e:
        return f"❌ Booking failed: {str(e)}"

    finally:
        cnx.close()

def cancel_ticket(pnr):
    cnx = connect_to_mysql()
    if not cnx:
        return "❌ DB connection failed"

    try:
        cursor = cnx.cursor(dictionary=True)
        cursor.execute("""
            SELECT fb.Status, fb.User_ID, fb.Flight_ID, fb.Travel_Date, fb.Booking_Date,
                   u.Name, fd.Departure_City, fd.Arrival_City
            FROM flight_booking_data fb
            JOIN user_data u ON fb.User_ID = u.ID
            JOIN flight_data fd ON fb.Flight_ID = fd.Flight_Number
            WHERE fb.PNR = %s
        """, (pnr,))
        result = cursor.fetchone()

        if not result:
            return f"❌ No booking found for PNR: {pnr}"

        if result["Status"] == "Cancelled":
            return f"ℹ️ Booking for PNR {pnr} is already cancelled."

        # Cancel it
        cursor.execute("UPDATE flight_booking_data SET Status = 'Cancelled' WHERE PNR = %s", (pnr,))
        cnx.commit()

        return (
            f"✅ Booking has been successfully cancelled for {result['Name']}.\n"
            f"🔐 PNR: {pnr}\n"
            f"🛫 Departure: {result['Departure_City']}\n"
            f"🛬 Arrival: {result['Arrival_City']}\n"
            f"🗓️ Travel Date: {result['Travel_Date']}\n"
            f"✈️ Flight Number: {result['Flight_ID']}\n"
            f"📌 Status: Cancelled\n\n"
            f"💳 A cancellation confirmation email will be sent to your registered email address with all the details.\n "
        )

    except Exception as e:
        return f"❌ Cancellation failed: {str(e)}"

    finally:
        cursor.close()
        cnx.close()