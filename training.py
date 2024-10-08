def train_vae(vae, data, epochs=100, batch_size=8):
    history = vae.fit(data, data, epochs=epochs, batch_size=batch_size, validation_split=0.15)
    return history

history = train_vae(vae, normalized_data, epochs=100, batch_size=32)

# Save the full VAE model
vae.save('vae_model.h5')

# Save the encoder model
encoder.save('vae_encoder.h5')

# Save the decoder model
decoder.save('vae_decoder.h5')

# Plotting the losses
plt.figure(figsize=(12, 4))
# Plot KL loss
plt.subplot(1, 3, 1)
plt.plot(history.history['kl_loss'], label='KL Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('KL Loss')
plt.legend()

# Plot VAE loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='VAE Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VAE Loss')
plt.legend()

# Plot Reconstruction loss
plt.subplot(1, 3, 3)
plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Reconstruction Loss')
plt.legend()
plt.tight_layout()
plt.show()
