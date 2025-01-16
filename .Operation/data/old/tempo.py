from datetime import datetime, timedelta

# Define o horário inicial, o Delta T (em minutos) e o horário limite
horario_inicial = datetime.strptime("00:00", "%H:%M").time()
delta_t_minutos = 30  # Delta T de 30 minutos
limite_horario = datetime.strptime("23:59", "%H:%M").time()

# Converte o horário inicial para um objeto datetime para facilitar a manipulação
horario_atual = datetime.combine(datetime.today(), horario_inicial)

# Lista para armazenar os horários
vetor_horarios = []

# Loop para criar o vetor de horários
while horario_atual.time() < limite_horario:
    vetor_horarios.append(horario_atual.time())
    # Adiciona Delta T ao horário atual
    horario_atual += timedelta(minutes=delta_t_minutos)

# Exibe o vetor de horários
for horario in vetor_horarios:
    print(horario.strftime("%H:%M"))