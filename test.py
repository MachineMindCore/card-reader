import card_reader.analizer as cr
from icecream import ic

if __name__ == "__main__":
    TEST_IMAGE = "images/card_A.jpeg"

    encoded = cr.encode_image(TEST_IMAGE)
    full_image = cr.decode_image(encoded)
    card_image = cr.extract_card(full_image)
    card_text = cr.extract_text(card_image)

    print(card_text)
