from configparser import ConfigParser
import os
import psycopg2

DEBUG = False

def db_config(filename='database.ini', section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    dir = os.path.dirname(os.path.realpath(__file__))
    parser.read(dir + os.sep + filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


def establish_db_conn(filename='database.ini', section='postgresql'):
    #dir = os.path.dirname(os.path.realpath(__file__))
    """ Connect to the PostgreSQL database server """  
    conn = None
    try:
        # read connection parameters
        params = db_config()

        # connect to the PostgreSQL server
        if DEBUG: print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        #print(conn)
        #conn = psycopg2.connect(host="localhost",database="ebird", 
        #                               user="postgres", password="admin")

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        if DEBUG: print('PostgreSQL database version:')
        cur.execute('SELECT version()')
        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        if DEBUG: print(db_version)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)

    finally:
        return conn

def terminate_db_conn(conn):
    if conn is not None:
        try:
            conn.close()
            if DEBUG: print('Database connection closed.')
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

## STAND-ALONE SCRIPT TEST
'''
def main():
    conn = establish_db_conn()

if __name__ == '__main__':
    main()
'''