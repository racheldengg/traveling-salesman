import sqlite3

def insert_optimal_solution(file_path):
    conn = sqlite3.connect('../../tsp.db')
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM complete_data")
    except sqlite3.ProgrammingError:
        print("Cursor is not connected.")
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS optimal_solutions (filename TEXT, length REAL);''')

    with open(file_path, "r") as file:
        lines = file.readlines()

        # Print each line
        for line in lines:
            line = line.strip()
            line = line.split(' : ')
            try:
                optimal_tour = int(line[1])
                instance = line[0]

                cursor.execute('''INSERT INTO optimal_solutions (filename, length) VALUES (?, ?)''',
                           (instance, optimal_tour))
            except:
                continue
    conn.commit()
    conn.close()


insert_optimal_solution('/home/rachel/Desktop/tsp-part-2/traveling-salesman/backend/data/txt_tsp_optimal/tsp_optimal.txt')