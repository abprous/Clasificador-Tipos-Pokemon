from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
import sys
import os

import requests # request img from web
import shutil # save img locally

from PIL import Image
from PIL import ImageOps

def resize_img(filename):
    img = Image.open(filename).convert('RGBA')
    new_image = Image.new("RGBA", img.size, "WHITE")
    new_image.paste(img, (0, 0), img)
    img = ImageOps.pad(new_image, (300,300), centering=(0.5,0.5), color=(255, 255, 255))

    os.remove(filename)
    filename = filename.rsplit('.', 1)[0] + '.jpg'
    img.convert('RGB').save(filename)

def save_img(curr_pokemon, curr_pokemon_img):
    img_url = curr_pokemon.find_element(By.TAG_NAME, "img").get_attribute("src")
    curr_pokemon_img = f"{curr_pokemon_img}.{img_url.rsplit('.', 1)[-1]}"
    res = requests.get(img_url, stream = True)

    if res.status_code == 200:
        with open(curr_pokemon_img,'wb') as f:
            shutil.copyfileobj(res.raw, f)
    else:
        print('Image Couldn\'t be retrieved')

    resize_img(curr_pokemon_img)

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
    driver.get('https://pokemondb.net/pokedex/national')

    pokemon_img_path = f"{exec_path}images/"

    if not os.path.exists(pokemon_img_path):
        os.mkdir(pokemon_img_path)

    print("Waiting for the page")
    time.sleep(2)

    gen_infocards = driver.find_elements(By.CLASS_NAME, "infocard-list")

    for gen_num, gen_elem in enumerate(gen_infocards):
        print(f"Retrieving pokemon images of gen {gen_num+1}")

        pokemon_elems = gen_elem.find_elements(By.CLASS_NAME, "infocard")

        for curr_pokemon in pokemon_elems:
            pokemon_name = curr_pokemon.find_element(By.CLASS_NAME, "ent-name").text.replace(":", "")
            print(f"Current pokemon is {pokemon_name}")

            curr_pokemon_path = f"{pokemon_img_path}{pokemon_name}/"

            info_url = curr_pokemon.find_element(By.TAG_NAME, "a").get_attribute("href")

            # Open new tab
            driver.switch_to.new_window()

            driver.get(f"https://pokemondb.net/{info_url}")


            multiple_variants = driver.find_elements(By.CSS_SELECTOR, ".tabset-basics > .sv-tabs-tab-list > a")
            multiple_variants_names = [elem.text for elem in multiple_variants]

            #if (multiple_variants):
            #    variant_elems = multiple_variants.find_elements(By.TAG_NAME, "a")


            driver.get(f"https://pokemondb.net/{info_url.replace('pokedex', 'artwork')}")

            artwork_list = driver.find_elements(By.CLASS_NAME, "grid-col")

            if (not artwork_list):
                driver.get(f"https://pokemondb.net/{info_url.replace('artwork', 'pokedex')}")

                if (len(multiple_variants) == 1):
                    if not os.path.exists(curr_pokemon_path):
                        os.mkdir(curr_pokemon_path)

                    curr_pokemon = driver.find_element(By.CLASS_NAME, "grid-col")
                    curr_pokemon_img = f"{curr_pokemon_path}0"
                    save_img(curr_pokemon, curr_pokemon_img)
                else:
                    for variant in multiple_variants:
                        variant_name = variant.text

                        curr_variant_path = f"{pokemon_img_path}{variant_name}/"
                
                        if not os.path.exists(curr_variant_path):
                            os.mkdir(curr_variant_path)
                        
                        variant.click()

                        time.sleep(1)

                        curr_pokemon = driver.find_element(By.CLASS_NAME, "grid-col")
                        
                        curr_pokemon_img = f"{curr_variant_path}0"
                        save_img(curr_pokemon, curr_pokemon_img)

            else:
                if (len(multiple_variants) == 1):
                    if not os.path.exists(curr_pokemon_path):
                        os.mkdir(curr_pokemon_path)
                    for num, curr_artwork in enumerate(artwork_list):
                        curr_pokemon_img = f"{curr_pokemon_path}{num}"
                        save_img(curr_artwork, curr_pokemon_img)
                else:
                    num_img = [[name, 0] for name in multiple_variants_names]
                    num_img.reverse()
                    for curr_artwork in artwork_list:
                        variant_text = curr_artwork.find_element(By.CLASS_NAME, "text-muted").text

                        for curr_num in num_img:
                            curr_variant = curr_num[0]
                            if (curr_variant in variant_text):
                                curr_variant_path = f"{pokemon_img_path}{curr_variant}/"

                                if not os.path.exists(curr_variant_path):
                                    os.mkdir(curr_variant_path)

                                curr_pokemon_img = f"{curr_variant_path}{curr_num[1]}"
                                save_img(curr_artwork, curr_pokemon_img)

                                curr_num[1] += 1

                                break
                                



            driver.close()
            driver.switch_to.window(driver.window_handles[0])


    return

if __name__ == "__main__":
    main()
