# RL residual em GPU (MuJoCo Warp)

O controlador model-based (LQR + VMC + PIDs, [`control/`](../../control/)) continua equilibrando o robô.
A política de RL só soma um **residual** em pontos definidos, e só no modo `operating`. Com ação
zero, o robô é exatamente o controlador original. O treino roda no **MuJoCo Warp** (GPU), com o mesmo
modelo do CAD em que o LQR foi validado.

```text
política (50 Hz)                          ação ∈ [-1, 1]^6 × escala
  [dL_xp, dL_xm]   comprimento de cada perna (média → altura; diferença → roll do corpo)
  [dth_xp, dth_xm] ângulo de cada perna (média → inclina o equilíbrio do LQR; diferença → abre uma perna)
  [dT_xp, dT_xm]   torque somado em cada roda
        │
LQR + PIDs + modo automático do trem   (kernel Warp, 1 kHz, uma thread por robô)
        │
VMC: jacobiano analítico do 5 barras → torques dos 4 quadris + 2 rodas
        │
MuJoCo Warp (1 kHz), milhares de robôs no mesmo terreno
```

## Arquivos

| arquivo | o que faz |
|---|---|
| `model.py` | modelo de treino: o MJCF do CAD com colisão simplificada (pneus cilíndricos + pontas do trem) e o terreno como heightfield |
| `terrain.py` | terreno grande compartilhado, em grade (linhas = dificuldade 0–5; colunas = plano, irregular, rampa, ondas, degraus, blocos) |
| `controller.py` | o controlador de `control/repo_control_policy.py` portado para um kernel Warp, mais o residual |
| `env.py` | ambiente RSL-RL: reset, observações, recompensas, terminação, curriculum, empurrões, randomização |
| `train.py` | PPO (RSL-RL 5.4.1); cria `experiments/<data>-warp-residual/`; ao final roda `eval.py` e `render.py` |
| `eval.py` | só o LQR × LQR + RL nos mesmos cenários: quedas por minuto e erro de velocidade por terreno |
| `render.py` | `video_overview.mp4` (vários robôs no mesmo terreno) e `video_compare.mp4` (LQR em cima, LQR + RL embaixo) |
| `tests/test_controller_parity.py` | compara o kernel Warp com o controlador numpy, passo a passo e em malha fechada |

## Rodar

```bash
cd PRIMUS
~/test/isaac-rl/bin/python sim/warp_rl/tests/test_controller_parity.py      # kernel == controlador numpy
~/test/isaac-rl/bin/python sim/warp_rl/train.py --num-envs 4096 --iterations 2000
~/test/isaac-rl/bin/python sim/warp_rl/eval.py --checkpoint <run>/logs/model_1999.pt --out <run>
~/test/isaac-rl/bin/python sim/warp_rl/render.py --checkpoint <run>/logs/model_1999.pt --out <run>
```

Usa o ambiente Python compartilhado `~/test/isaac-rl` (MuJoCo 3.11, MuJoCo Warp 3.11, Warp 1.16,
torch cu130, RSL-RL 5.4.1), sem instalar nada. Na RTX 5070, com 4096 robôs, são ~45 mil passos de
política por segundo (~2,2 s por iteração de PPO).

## Decisões e o porquê

- **Residual em vez de torque bruto.** O LQR já equilibra. O RL começa do controlador model-based
  (última camada do actor zerada) e só aprende correções. Isso evita o "sentar no trem de pouso" do
  treino anterior no Isaac.
- **Residual no espaço da perna, por perna.** Permite pernas desalinhadas (uma à frente, uma mais
  curta) sem o RL ter de aprender a cinemática do 5 barras.
- **Física e controlador num grafo CUDA.** São 20 passos de física de 1 ms por passo de política, com
  o controlador rodando a cada passo, como no MuJoCo de CPU. O kernel custa ~16 µs por passo para
  4096 robôs.
- **Colisão mínima.** No MuJoCo Warp, a colisão com o heightfield testa cada peça contra todas as
  células embaixo dela, e o custo cresce rápido. Por isso só os pneus e as pontas do trem colidem
  com o terreno. A queda é detectada pela geometria (inclinação > 50°, chassi < 0,2 m acima do chão,
  proteção do controlador). O heightfield tem 8 cm de resolução, e a quina de um degrau vira uma
  rampa de 8 cm.
- **Actor só com o que o robô mede:** IMU, comprimento e ângulo das pernas (encoders dos quadris),
  velocidade das rodas, odometria, comandos e 5 quadros de histórico, com ruído. O critic também
  recebe a velocidade real, a altura sobre o chão, um mapa de alturas e os parâmetros sorteados.
- **Randomização:** atrito de 0,5 a 1,25, massa do chassi ±15%, CoM ±2 cm, empurrões de até
  0,6 m/s a cada 3–7 s, e 15% dos episódios começando estacionado.
- **Torque e massa** ficam nos valores atuais do modelo (18 N·m na junta, 15,7 kg) até a
  caracterização real.

## Limitações conhecidas

- Sem câmera: obstáculos maiores que ~5 cm e escadas de verdade ficam para a próxima fase (Isaac,
  mapa de elevação).
- O residual não age durante levantar e pousar; esses modos são só o controlador model-based.
- O contato pneu × heightfield gera 1 ponto por pneu (limitação do MuJoCo Warp para cilindro × heightfield).
