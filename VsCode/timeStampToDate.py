import time

def to_date(df):
    time_type = []
    date_type = []

    for i in range(0, len(df)):
        t_type = time.strftime("%X", time.localtime(df["time"].iloc[i]))
        d_type = time.strftime("%D", time.localtime(df["time"].iloc[i]))
        time_type.append(t_type)
        date_type.append(d_type)

    df["time_t"] = time_type
    df["date"] = date_type
    

    return df
