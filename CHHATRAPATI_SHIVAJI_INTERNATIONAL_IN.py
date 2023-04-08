import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
K.set_epsilon(1e-8)
# Read and preprocess data
df = pd.read_csv('CHHATRAPATI SHIVAJI INTERNATIONAL, IN.csv', usecols=['DATE','REPORT_TYPE', 'WND'])
df[['Angle', 'Angle_Measurement_Quality', 'Wind_Obs_Character', 'Wind_Speed', 'Wind_Speed_Quality']] = df['WND'].str.split(",", expand=True)

df = df.astype({'Angle': float, 'Angle_Measurement_Quality': float, 'Wind_Obs_Character': str, 'Wind_Speed': float, 'Wind_Speed_Quality': float})

df = df[(df['Angle'] != 999) & (df['Angle_Measurement_Quality'] == 1) & (df['Wind_Obs_Character'] == 'N') & (df['Wind_Speed'] != 9999) & (df['Wind_Speed_Quality'] == 1) & (df['REPORT_TYPE'] == 'FM-15')]

df['DATE'] = pd.to_datetime(df['DATE'])
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month
df['Day'] = df['DATE'].dt.day
df['Hour'] = df['DATE'].dt.hour
df['Minutes'] = df['DATE'].dt.minute
df['Seconds'] = df['DATE'].dt.second

df = df[['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds','REPORT_TYPE', 'Wind_Speed', 'Angle']]
df = df[(df['Minutes'] == 0) & (df['Seconds'] == 0)]
df.to_csv('Modified_CHHATRAPATI SHIVAJI INTERNATIONAL, IN.csv', index=False)
# Create a DataFrame with hourly timestamps from 2012 to 2022
date_range = pd.date_range(start='2018-01-01', end='2022-12-31', freq='H')
df1 = pd.DataFrame(date_range, columns=['Date'])
df1[['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds']] = df1.apply(lambda x: [x.Date.year, x.Date.month, x.Date.day, x.Date.hour, x.Date.minute, x.Date.second], axis=1, result_type="expand")

merged_df = pd.merge(df1, df, on=['Year', 'Month', 'Day', 'Hour', 'Minutes', 'Seconds'], how='outer')
merged_df[['REPORT_TYPE', 'Wind_Speed', 'Angle']] = merged_df[['REPORT_TYPE', 'Wind_Speed', 'Angle']].fillna(value=pd.NA)

wind_speed_missing_pct = merged_df['Wind_Speed'].isna().mean() * 100
angle_missing_pct = merged_df['Angle'].isna().mean() * 100

print(f"Percentage of missing values in Wind_Speed column: {wind_speed_missing_pct:.2f}%")
print(f"Percentage of missing values in Angle column: {angle_missing_pct:.2f}%")

merged_df.to_csv('Merged_Modified_CHHATRAPATI SHIVAJI INTERNATIONAL, IN.csv', index=False)

from tensorflow.keras import layers
import numpy as np

# Define the GAN architecture
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, input_dim=100, kernel_initializer='he_normal'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(64, kernel_initializer='he_normal'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(2, activation='linear'))
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, input_dim=2, kernel_initializer='he_normal'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(64, kernel_initializer='he_normal'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

# Define the training loop
@tf.function
def train_step(generator_model, discriminator_model, real_data):
    available_data = real_data[:, :2]
    missing_indices = tf.math.reduce_any(tf.math.is_nan(available_data), axis=1)
    
    noise = tf.random.normal([len(available_data), 100])
    generated_data = generator_model(noise)
    generated_data = tf.cast(generated_data, dtype=tf.float32)  # Cast generated_data to float32
    available_data = tf.cast(available_data, dtype=tf.float32)  # Cast available_data to float32
    available_data = tf.where(tf.math.is_nan(available_data), generated_data, available_data)

    fake_data = tf.concat([available_data[~missing_indices], generated_data], axis=0)  # Create fake_data

    with tf.GradientTape() as disc_tape:
        real_output = discriminator_model(available_data, training=True)
        fake_output = discriminator_model(generated_data, training=True)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator_model.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

    with tf.GradientTape() as gen_tape:
        generated_data = generator_model(noise, training=True)
        fake_output = discriminator_model(generated_data, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    return disc_loss, gen_loss, missing_indices

# Train the GAN
generator_model = make_generator_model()
discriminator_model = make_discriminator_model()

available_data = merged_df[['Wind_Speed', 'Angle']].values

missing_indices = np.isnan(available_data).any(axis=1)

# for epoch in range(1000):
for epoch in range(201):
    for i in range(100):
        real_data = available_data[np.random.choice(len(available_data), size=128, replace=False)]
        disc_loss, gen_loss, _ = train_step(generator_model, discriminator_model, real_data)


    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Disc loss: {disc_loss.numpy()}, Gen loss: {gen_loss.numpy()}')


# Generate predictions for the missing data
missing_count = np.sum(missing_indices)
noise = tf.random.normal([missing_count, 100])
generated_data = generator_model(noise).numpy()
available_data[missing_indices] = generated_data

# Update the merged_df DataFrame
merged_df.loc[:, 'Wind_Speed'] = available_data[:, 0]
merged_df.loc[:, 'Angle'] = available_data[:, 1]

# Define datetime for saving the updated DataFrame
from datetime import datetime
# Save the updated DataFrame to a CSV file with current timestamp
merged_df.to_csv(f'Updated_CHHATRAPATI_SHIVAJI_INTERNATIONAL_IN_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv', index=False)
# merged_df.to_csv('Updated_CHHATRAPATI_SHIVAJI_INTERNATIONAL_IN_1.csv', index=False)