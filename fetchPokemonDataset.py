from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import sys
import os


def main():
    
    os.system('cls')

    global exec_path

    exec_path = sys.argv[0]
    exec_path = exec_path.replace("\\", "/")

    i_last_slash = exec_path[::-1].find("/")

    exec_path = exec_path[: -i_last_slash]

    print(f"Path: {exec_path}")

    chrome_service = Service(f"{exec_path}/chromedriver.exe")
    chromium_options = webdriver.ChromeOptions()

    try:
        driver = webdriver.Chrome(service=chrome_service, options=chromium_options)
    except:
        print("Chrome Driver not in same directory, place it correctly!")
        exit()
    
    driver.maximize_window()
    driver.get('https://pokemondb.net/pokedex/all')

    csv_file = open(f"{exec_path}Pokedex_Cleaned.csv", "wb")
    csv_file.write(("#,name,type_1").encode('utf-8'))
    csv_file.write(('\n').encode('utf-8'))

    print("Waiting for the page")
    time.sleep(2)

    primary_table = driver.find_element(By.ID, "pokedex")
    pokemon_list = primary_table.find_elements(By.CSS_SELECTOR, "tbody > tr")


    for curr_pokemon in pokemon_list:
        current_line = []

        pokemon_num = int(curr_pokemon.find_element(By.CLASS_NAME, "infocard-cell-data").text)
        current_line.append((str(pokemon_num) + ",").encode('utf-8'))

        pokemon_name = curr_pokemon.find_elements(By.TAG_NAME, "small")

        if (not pokemon_name):
            pokemon_name = curr_pokemon.find_element(By.CLASS_NAME, "ent-name").text.replace(":", "")
        else:
            pokemon_name = pokemon_name[0].text. replace(":", "")

        print(f"Current pokemon is {pokemon_name}")

        current_line.append((pokemon_name + ",").encode('utf-8'))

        current_type = curr_pokemon.find_element(By.CLASS_NAME, "type-icon").text
        current_type = current_type[0].upper() + current_type[1:].lower()
        current_line.append((current_type).encode('utf-8'))

        csv_file.writelines(current_line)
        csv_file.write(('\n').encode('utf-8'))

    csv_file.close()

    return

if __name__ == "__main__":
    main()
