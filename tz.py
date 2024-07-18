import datetime
import pytz

def get_tz():
    # Get the timezone from the system
    tz = datetime.datetime.now(pytz.timezone('UTC')).astimezone().tzinfo
    # Return the timezone in the format like "Asia/Kolkata"
    return str(tz)

timezone = get_tz()
print("Timezone:", timezone)
