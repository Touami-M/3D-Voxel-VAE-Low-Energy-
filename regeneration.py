def regenerate_samples(decoder, num_samples, scaler, input_shape):
    # Generate latent space samples
    latent_samples = np.random.normal(size=(num_samples, 32))
    
    # Regenerate data from latent space
    regenerated_data = decoder.predict(latent_samples)
    regenerated_data=np.squeeze(regenerated_data)
    
    return regenerated_data

num_samples = 10000
regenerated_samples = regenerate_samples(decoder, num_samples, scaler, input_shape)
