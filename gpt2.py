from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo y el tokenizador
model_name = "datificate/gpt2-small-spanish"  # Modelo en español de GPT-2
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Configuración del modelo
model.config.pad_token_id = tokenizer.eos_token_id  # Asegurar que el token de padding es el token de fin de secuencia

# Longitud máxima de entrada y respuesta
MAX_INPUT_LENGTH = 512  # Ajusta según tus necesidades
MAX_OUTPUT_LENGTH = 150  # Longitud máxima de la respuesta

def generate_response(prompt):
    # Tokenizar la entrada
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LENGTH, padding=True)
    input_ids = inputs["input_ids"]

    # Generar la respuesta
    outputs = model.generate(
        input_ids,
        max_length=MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH,  # Longitud total máxima
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,  # Usar el modo de muestreo
        temperature=0.7,  # Controlar la creatividad
        top_k=50,  # Top-k sampling
        top_p=0.95  # Nucleus sampling
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Hola, ¿en qué puedo ayudarte? (Escribe 'salir' para terminar la conversación)")

    while True:
        user_input = input("Tú: ")
        
        # Comando para salir del chat
        if user_input.lower() in ['salir', 'eso es todo']:
            print("Bot: OK, que tengas una buena tarde.")
            break

        # Crear el prompt para el modelo
        prompt = f"Responde en español: {user_input}"
        response = generate_response(prompt)
        print("Bot:", response)

if __name__ == "__main__":
    main()
