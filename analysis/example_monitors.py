

def detectExtremeValues(row):
    if abs(row['angular_speed_z']) > 0.2:
        return True
    return False