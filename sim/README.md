# PRIMUS — simulação (MuJoCo + Isaac Sim)

Simulação do robô a partir da montagem **"Robo bipedal - montagem principal"** do Fusion: malhas, massas e inércias do CAD, pernas 5 barras fechadas, atuadores 9:1. O controlador (LQR + VMC + PIDs em cascata, estrutura do `Skythinker616/foc-wheel-legged-robot`) fica em [`../control/`](../control/), compartilhado com o robô real; por padrão ele usa **só o que o robô real mede** (IMU, encoders e torque dos motores).

Todos os comandos abaixo rodam **de dentro de `sim/`**.

## Estrutura

```text
models/robo_bipedal.xml                 robô (GERADO): corpos, juntas, laços, atuadores, sensores
models/cena_arena.xml                   cena "arena": terreno, rampa, escada, pisos, lombadas, pedras
models/cena_plana.xml                   cena plana (para medir o controle)
models/meshes/                          malhas visuais e de colisão (do CAD)
models/cad/                             massas, inércias e listas de malhas exportadas do CAD
models/assets/terrain_heightmap.png     relevo da arena
../control/repo_control_policy.py       controlador (LQR + VMC + PIDs) + IMU simulada
../control/lqr_design.py                recalcula os ganhos do LQR a partir do MJCF
../control/lqr_gains.py                 tabela de ganhos (GERADA)
scripts/robot_model.py                  carregar modelo/keyframe, cinemática da perna, parâmetros do LQR
scripts/generate_mjcf.py                gera os 3 XML a partir de models/cad + tools/robot_def.py
scripts/generate_urdf.py                gera isaac/urdf/robo_bipedal.urdf + models/cad/loop_closures.json
isaac/                                  URDF + scripts do Isaac Sim / Isaac Lab (laços fechados no USD)
scripts/run_viewer.py                   viewer com comando zero
scripts/run_xbox_viewer.py              viewer controlado pelo controle de Xbox
scripts/xbox_controller.py, inspect_gamepad.py
scripts/check_model.py                  carrega, lista juntas/atuadores, equilibra 5 s, mede o laço
scripts/drive_demo.py                   demo headless andando
scripts/diagnose.py                     log de trajetória para ajuste
scripts/analyze_balance.py              métricas de balanço (pitch, quique, contato, torque, deriva)
scripts/record_video.py                 grava um mp4 headless na arena
tools/                                  pipeline CAD → malhas/massas (scripts do Fusion + build_assets.py)
../docs/notas_modelo_controle.md                  modelo, convenções, controle, o que é provisório
../docs/demo_arena.mp4                  vídeo de validação
```

## Rodar

Use o env conda `bipedal`, como antes:

```bash
cd ~/Desktop/primus/sim
conda run -n bipedal python3 -m pip install -r requirements.txt    # se faltar algo

conda run -n bipedal python3 scripts/check_model.py               # teste rápido
conda run -n bipedal mjpython scripts/run_viewer.py               # viewer (macOS: mjpython)
conda run -n bipedal mjpython scripts/run_xbox_viewer.py          # com o controle de Xbox
conda run -n bipedal mjpython scripts/run_xbox_viewer.py --flat   # chão plano
```

### Controle de Xbox

| entrada | ação |
|---|---|
| analógico esquerdo, eixo Y | velocidade (até 1,5 m/s; ajuste com `--max-speed`) |
| analógico direito, eixo X | taxa de giro (yaw) |
| D-pad ↑/↓ ou gatilhos | altura; **trava** no valor. Perna de 0,18 a 0,36 m (nominal 0,27) |
| A | parada suave |
| B | freio forte |
| X | trem de pouso: abaixado ↔ recolhido |
| Start | reseta alvos e altura |

Se o mapeamento for diferente: `python3 scripts/inspect_gamepad.py`.

### Medir

```bash
conda run -n bipedal python3 scripts/analyze_balance.py --flat --speed 0.6 --yaw 1.5
conda run -n bipedal python3 scripts/analyze_balance.py --speed 0.4 --yaw 1.0            # arena
conda run -n bipedal python3 scripts/analyze_balance.py --flat --imu mahony --imu-noise  # IMU realista
conda run -n bipedal python3 scripts/drive_demo.py --seconds 6 --target-speed 0.5 --yaw 0.3
conda run -n bipedal python3 scripts/record_video.py ../docs/demo_arena.mp4                 # precisa de ffmpeg
```

## Estado do robô: sensores do robô real (padrão) ou estado do simulador

`RepoControlPolicy(..., state_source="robot")` é o padrão e usa só o que o robô de verdade tem:

| grandeza | de onde vem |
|---|---|
| pitch, roll, yaw e taxas | IMU simulada: sensores `imu_quat`, `imu_gyro`, `imu_acc` no lugar do WitMotion |
| x, dx (para frente) | encoders das rodas, **corrigidos pelo giro da canela** (giro da base no IMU + cinemática dos quadris). O projeto antigo usava só a média dos encoders, o que mistura o balanço da perna com a velocidade |
| roda no chão | força normal **estimada** pelo torque dos motores do quadril + aceleração da roda (como no firmware de referência) |
| geometria da perna | cinemática dos quadris (= encoders) |

Com `state_source="sim"` (ou `--state sim` nos scripts) o controlador usa a velocidade real do centro das rodas e a força real em cada pneu. Serve para comparar e como observação privilegiada do *critic* no RL (o *actor* deve ver só o que o robô tem).

Para sensores mais realistas, `ImuConfig` tem ruído e bias de giro e o modo `mahony`, que estima a atitude só com giro e acelerômetro. Veja `../docs/notas_modelo_controle.md`.

## Isaac Sim / primeiro teste de RL

O fluxo atualizado para Isaac Sim 6.1 e Isaac Lab v3.0.0-EA está em [isaac/README.md](isaac/README.md). Inclui diagnóstico do runtime compartilhado, conversão URDF → USD com fechamento dos laços, teste físico e tarefa inicial de equilíbrio com PPO. A conversão real, o teste PhysX com dois robôs e um PPO de 20 iterações foram executados em CPU. O primeiro treino de terreno em GPU também foi concluído: 3.000 atualizações com 128 ambientes, avaliação e vídeos. O desempenho em escadas e o rastreamento de comandos ainda precisam melhorar; veja [os resultados](../docs/terrain-run-2026-10-31.md).

## Regenerar

Se mudar algum parâmetro de atuador (o topo de `scripts/generate_mjcf.py`):

```bash
python3 scripts/generate_mjcf.py
python3 ../control/lqr_design.py --write     # massas/inércias mudaram? recalcula o LQR
```

Se o CAD mudar:

1. No Fusion, rode `tools/fusion/fusion_1_classificar_corpos.py` e `fusion_2_exportar_malhas.py`. Eles exportam os STL brutos para `~/Desktop/robo_sim_export/raw` (pasta temporária; pode apagar depois do passo 2).
2. Depois:
   ```bash
   pip install trimesh fast-simplification coacd
   RAW=~/Desktop/robo_sim_export/raw python3 tools/build_assets.py        # escreve models/meshes e models/cad
   python3 scripts/generate_mjcf.py && python3 scripts/generate_urdf.py && python3 ../control/lqr_design.py --write
   ```

