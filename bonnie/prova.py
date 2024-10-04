import cv2

def highlight_pixel(image_path, x, y):
    # Carica l'immagine
    img = cv2.imread(image_path)

    # Verifica che l'immagine sia stata caricata correttamente
    if img is None:
        print("Errore: Immagine non trovata!")
        return

    # Disegna un cerchio rosso attorno al pixel (x, y)
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # (0, 0, 255) è il rosso in BGR

    # Mostra l'immagine con il pixel evidenziato
    cv2.imshow("Immagine con pixel evidenziato", img)

    # Attende che si prema un tasto qualsiasi per chiudere la finestra
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Parametri: percorso immagine, coordinate pixel (x, y)
image_path = 'static/src_img.png'  # Inserisci il percorso dell'immagine qui
x, y = 359, 281  # Coordinate del pixel da evidenziare

highlight_pixel(image_path, x, y)
