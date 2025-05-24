import datetime
from math import sin, cos, tan, radians, degrees, acos, asin


class Sun:
    """Calculate sunrise and sunset based on equations from NOAA:
    http://www.srrb.noaa.gov/highlights/sunrise/calcdetails.html
    A similar implementation is available at
    https://michelanders.blogspot.com/2010/12/calulating-sunrise-and-sunset-in-python.html
    The algorithm is based on the Book Astronomical Algorithms by Jean Meeus.
    """

    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude

    @staticmethod
    def percentday2time(x):
        hours = 24 * x
        h = int(hours)
        minutes = (hours - h) * 60
        m = int(minutes)
        seconds = (minutes - m) * 60
        s = int(seconds)
        return datetime.time(h, m, s)

    @staticmethod
    def day_time_timezone_from_dateteime(date):
        # OpenOffice day zero is December 30, 1899
        day = date.toordinal() - datetime.datetime(1899, 12, 30).toordinal()
        t = date.time()
        time = (t.hour + t.minute / 60 + t.second / 3600) / 24
        offset = date.utcoffset()
        timezone = offset.total_seconds() / 3600 if offset else 0
        return day, time, timezone

    def get_suntimes(
        self, date: datetime.datetime
    ) -> tuple[datetime.time, datetime.time]:
        """Calculate sunrise and sunset times for a given date.

        The sunrise and sunset results are accurate to one minute for latitudes
        between +/- 72°, and 10 minutes outside those.  Calculations are valid
        for dates between 1901 and 2099, due to an approximation used in the
        Julian Day calculation.

        Args:
            date: The date for which to calculate sunrise and sunset.

        Returns:
            tuple: A tuple containing sunrise and sunset times in local time.
        """
        day, time, timezone = self.day_time_timezone_from_dateteime(date)

        julian_day = day + 2415018.5 + time - timezone / 24
        julian_century = (julian_day - 2451545) / 36525

        geom_mean_long_sun_deg = (
            280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032)
        ) % 360
        geom_mean_anom_sun_deg = 357.52911 + julian_century * (
            35999.05029 - 0.0001537 * julian_century
        )
        eccent_earth_orbit = 0.016708634 - julian_century * (
            0.000042037 + 0.0000001267 * julian_century
        )
        sun_eq_of_ctr = (
            sin(radians(geom_mean_anom_sun_deg))
            * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century))
            + sin(radians(2 * geom_mean_anom_sun_deg))
            * (0.019993 - 0.000101 * julian_century)
            + sin(radians(3 * geom_mean_anom_sun_deg)) * 0.000289
        )
        sun_true_long_deg = geom_mean_long_sun_deg + sun_eq_of_ctr
        sun_app_long_deg = (
            sun_true_long_deg
            - 0.00569
            - 0.00478 * sin(radians(125.04 - 1934.136 * julian_century))
        )
        mean_obliq_ecliptic_deg = (
            23
            + (
                26
                + (
                    (
                        21.448
                        - julian_century
                        * (
                            46.815
                            + julian_century * (0.00059 - julian_century * 0.001813)
                        )
                    )
                )
                / 60
            )
            / 60
        )
        obliq_corr_deg = mean_obliq_ecliptic_deg + 0.00256 * cos(
            radians(125.04 - 1934.136 * julian_century)
        )
        sun_declin_deg = degrees(
            asin(sin(radians(obliq_corr_deg)) * sin(radians(sun_app_long_deg)))
        )
        var_y = tan(radians(obliq_corr_deg / 2)) * tan(radians(obliq_corr_deg / 2))
        eq_of_time_minutes = 4 * degrees(
            var_y * sin(2 * radians(geom_mean_long_sun_deg))
            - 2 * eccent_earth_orbit * sin(radians(geom_mean_anom_sun_deg))
            + 4
            * eccent_earth_orbit
            * var_y
            * sin(radians(geom_mean_anom_sun_deg))
            * cos(2 * radians(geom_mean_long_sun_deg))
            - 0.5 * var_y * var_y * sin(4 * radians(geom_mean_long_sun_deg))
            - 1.25
            * eccent_earth_orbit
            * eccent_earth_orbit
            * sin(2 * radians(geom_mean_anom_sun_deg))
        )
        ha_sunrise_deg = degrees(
            acos(
                cos(radians(90.833))
                / (cos(radians(self.latitude)) * cos(radians(sun_declin_deg)))
                - tan(radians(self.latitude)) * tan(radians(sun_declin_deg))
            )
        )
        solar_noon_lst = (
            720 - 4 * self.longitude - eq_of_time_minutes + timezone * 60
        ) / 1440
        sunrise_time_lst = solar_noon_lst - ha_sunrise_deg * 4 / 1440
        sunset_time_lst = solar_noon_lst + ha_sunrise_deg * 4 / 1440

        sunrise_time_lst = self.percentday2time(sunrise_time_lst)
        sunset_time_lst = self.percentday2time(sunset_time_lst)

        return sunrise_time_lst, sunset_time_lst
