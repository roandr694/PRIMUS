# Notas do modelo e do controle (robô novo)

## 1. Modelo

- Origem: montagem "Robo bipedal - montagem principal" do Fusion. As malhas visuais são as do CAD, decimadas. A colisão usa cilindros exatos nos pneus (R = 0,1075 m) e cascas convexas (CoACD) no resto.
- O robô colide só com o chão e os obstáculos; não há autocolisão.
- Frames: **X lateral** (X+ = perna `xp` = direita), **Y frente**, **Z cima**. Todas as juntas giram em +X. O ângulo 0 de todas as juntas é a pose do CAD, com a perna quase esticada (L0 = 0,394 m).
- Nas cenas, o robô nasce girado −90° em Z, olhando para +X do mundo, que é onde a rampa da arena antiga está.

**Juntas e atuadores:**

- `hip_Yp_*` e `hip_Ym_*`: motores do quadril.
- `knee_Yp_*` e `knee_Ym_*`: joelhos passivos.
- `wheel_*`: rodas.
- `gear_front` e `gear_rear`: trem de pouso, com servo de posição.
- Cada perna é um 5 barras **fechado** por `equality/connect loop_*` no eixo da roda (entre `linkYm_*` e `shin_*`). A folga medida fica abaixo de 0,1 mm.

**Sites usados pelo controle:**

- `imu`: posição do WitMotion.
- `hip_mid_*`: quadril virtual, no meio dos dois quadris.
- `wheel_center_*`: centro da roda, preso na canela.
- `foot_*`: âncora do laço, no link Y−.

### Massas

O total é **15,68 kg**:

| corpo | kg |
|---|---|
| base (chassi, eletrônica, bateria, estatores do quadril) | 7,63 |
| canela, com o motor da roda (cada) | 1,35 |
| roda (cada) | 1,31 |
| link Y− (cada) | 0,65 |
| braço (cada um dos 4) | 0,33 |

As correções de material do CAD estão em `sim/tools/robot_def.py → MASS_OVERRIDES`:

- bateria M18;
- perfis de alumínio;
- Orin;
- lidar;
- RPi;
- aro da roda, **assumido em PETG — confirmar**.

### Atuadores: valores PROVISÓRIOS

Ficam em `sim/scripts/generate_mjcf.py`:

- redução real de 9:1;
- **torque de pico no motor = 2 N·m**, ou seja, 18 N·m na junta;
- armature de 0,006 kg·m².

No MJCF, `ctrl` é o torque **no motor**. O controlador calcula o torque na junta e divide por 9.

Para referência, nos testes:

| situação | quadril (junta) |
|---|---|
| parado em L0 = 0,27 m | ~6,5 N·m |
| parado em 0,20 m | ~5 N·m |
| parado em 0,34 m | ~9 N·m |
| mudando de altura rápido | encosta nos 18 N·m por ~0,2 s |

A roda raramente passa de 4 N·m. As lombadas de 7 cm da arena ficam no limite, porque a roda precisa empurrar ~2× o peso. Com 18 N·m na roda ele passa, mas devagar.

## 2. Controle (`control/repo_control_policy.py`)

A estrutura é a mesma do projeto antigo. O que mudou foi o que entra nela.

**LQR (balanço).** Estado `[θ, θ̇, x, ẋ, φ, φ̇]`, com u = −K(L0)·x:

- θ é o ângulo da perna em relação à vertical. É positivo quando o quadril está à frente da roda.
- φ é o pitch, positivo com o nariz para cima.
- T é o torque de UMA roda e Tp o torque virtual de UMA perna.
- K(L0) é um polinômio cúbico. Vem de `control/lqr_design.py`, um porte do `sys_calc.m`.
- O modelo é de **meio robô**, com parâmetros tirados do MJCF **em cada L0**. A perna é posicionada de verdade com o 5 barras fechado.

Pontos importantes desses parâmetros:

- o CoM da perna fica perto da roda (L ≈ 0,115 m em L0 = 0,27), porque o motor da roda está na canela;
- a inércia da perna, em torno do próprio CoM, é ~0,09 kg·m²; os braços são longos para frente e para trás;
- a inércia da roda inclui a armature do motor;
- o CoM do corpo fica 2,1 cm **abaixo** do quadril e 1,1 cm atrás.

Os pesos são Q = [10, 4, 40, 80, 600, 10] e R = [4, 4], com dt = 1 ms.

**Equilíbrio.** O CoM fora do eixo e as rampas são compensados por um integral lento do erro de posição, que ajusta o zero de θ (`theta_zero_ki`, limitado a ±0,15 rad).

**VMC.** Converte a força ao longo da perna F e o Tp nos torques dos 2 quadris, via τ = Jᵀ·F_mundo. J é o jacobiano do centro da roda em relação aos quadris, projetado pela restrição do laço, para que os joelhos passivos acompanhem.

- F = PID de comprimento + peso por perna (massa sem as rodas × g / 2) ± roll.
- Tp = LQR ± sincronismo das pernas. O sincronismo precisa de força: o diferencial de yaw abre as pernas.

**PIDs em cascata:**

- yaw: taxa com feedforward e correção lenta de heading;
- roll: nivelador P-D;
- comprimento da perna;
- ângulo entre as pernas.

Os ganhos foram ajustados para este robô e para dt = 1 ms.

**Trem de pouso.** `RobotCommand.gear_down` liga o servo: 0° é abaixado (pose do CAD), ±100° é recolhido.

- Abaixado, a ponta fica 9 cm acima do chão em L0 = 0,27 m. Por isso **L0 mínimo = 0,18 m**.
- Recolhido, a ponta sobe para 20 cm.

### Estado: sensores do robô real (padrão) ou simulador

`state_source="robot"` (padrão) usa só o que o robô real mede; `state_source="sim"` usa o estado perfeito do simulador.

| sinal | `robot` (padrão) | `sim` |
|---|---|---|
| atitude | `imu_quat` (ângulo já fundido, como o WitMotion); ou Mahony com giro + acelerômetro em `ImuConfig(mode="mahony")` | igual |
| φ̇, taxa de roll, taxa de yaw | `imu_gyro` | igual |
| x, ẋ | v = −R·(ω_encoder_roda + ω_canela), com ω_canela = giro X do IMU + derivada do ângulo da canela em relação à base (quadril + joelho, que sai da cinemática do 5 barras). Não vê escorregamento | velocidade real do centro das rodas no heading |
| contato | Fn estimada: torque medido dos 2 quadris → força da perna na roda (inverte τ = Jᵀf) + m_abaixo_do_quadril·(g + acel. vertical da roda), filtrada; no chão se > 15 N por mais de 30 ms | força real no pneu (`mj_contactForce`) > 5 N |
| L0, ângulo da perna | cinemática dos quadris | igual |

Medido (chão plano, andando e girando): erro RMS da velocidade pelos encoders ≈ 0,04 m/s (picos em freadas, quando a roda escorrega); Fn estimada vs real: viés ~1 N, RMS ~5 N. Num pulo, o estimador marca "no ar" em ~80% do tempo real no ar, sem falsos positivos com a roda no chão. O estimador começa assumindo "no chão" no `reset`, porque depende da própria força comandada.

Para RL: dê ao *actor* só as observações do modo `robot` e, se quiser, as do modo `sim` ao *critic* (actor-critic assimétrico).

Ruído opcional em `ImuConfig`: `gyro_noise`, `gyro_bias`, `angle_noise`, `acc_noise`. O `--imu-noise` do `sim/scripts/analyze_balance.py` usa 0,005 rad/s, 0,002 rad/s, 0,002 rad e 0,05 m/s².

## 3. Resultados (headless)

| teste | resultado |
|---|---|
| parado, chão plano | pitch estável em +1,7° e pico a pico < 0,2°; L0 = 0,270 m |
| velocidade 0,5 / 1,5 / −1,0 m/s | segue, com 10–20% de sobressinal na partida; freio forte para em ~1 s |
| yaw 1,0 rad/s parado e andando 0,5 m/s | 1,00 / 0,99 rad/s |
| 1 m/s + yaw ±1,5 rad/s, com troca de altura | sem queda; roll ~1,5° para fora da curva |
| empurrão de 80 N por 0,1 s | recupera em ~1 s |
| altura 0,20 → 0,34 m | segue |
| arena: rampa de 13° (sobe, plataforma, desce), pisos, terreno | sem queda |
| IMU Mahony com ruído e bias | sem queda; pitch pico a pico 0,5° |
| todos os testes acima | feitos no modo `robot` (encoders + IMU + contato estimado) e no modo `sim` |

## 4. Para fazer quando houver dados reais

1. Torque de pico e velocidade máxima dos motores: em `sim/scripts/generate_mjcf.py`, `HIP_MOTOR_TAU` e `WHEEL_MOTOR_TAU`.
2. Inércia do rotor: `*_ARMATURE`.
3. Massa real dos atuadores e material do aro da roda.

Depois de mudar qualquer um deles: `sim/scripts/generate_mjcf.py` e `control/lqr_design.py --write`.
