#!/usr/bin/env python
import argparse
import psycopg2
import sys

def main(args):
    #Define our connection string
    conn_string = "host='labeling2' dbname='tractruck_new' user='tractruck_viewer' password='tractruck_viewer'"

    # print the connection string we will use to connect

    # get a connection, if a connect cannot be made an exception will be raised here
    conn = psycopg2.connect(conn_string)

    # conn.cursor will return a cursor object, you can use this cursor to perform queries
    cursor = conn.cursor()
    # execute our Query
    SQL = \
    '''
        SELECT
            vehicle,
            timestamp,
            weather
        FROM t_weather_info
        WHERE
            timestamp >= 1687496543 AND
            timestamp < 1687496843 AND
            vehicle = 'pdb-l4e-b0008'
    '''
    cursor.execute(SQL)

    # retrieve the records from the database
    records = cursor.fetchall()
    print('len:{}'.format(len(records)))
    print('type:{}'.format(type(records[0])))
    print('tuple len:{}'.format(len(records[0])))
    print('type:{}'.format(type(records[0][2])))
    print('dict len:{}'.format(len(records[0][2])))
    print('humidity:{}'.format(records[0][2]['humidity']))
    print(records)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('weather query')
    
    args = parser.parse_args()
    main(args)