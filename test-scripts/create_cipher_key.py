from cryptography.fernet import Fernet

key = Fernet.generate_key()
print(key.decode())  # save this string securely (e.g. env var, secrets manager)
