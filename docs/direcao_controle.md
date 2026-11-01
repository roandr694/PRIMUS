# Direção do controle do PRIMUS

*Atualizado em 2026-10-31.*

O controle é feito em camadas. O controlador **model-based** (LQR + VMC + PIDs) é a base: ele equilibra,
anda, gira, muda de altura e pousa, e é o que vai rodar primeiro no robô real. O **RL** fica por cima,
como uma camada de robustez. Ele soma correções pequenas ao model-based para empurrões, terreno estranho
e pernas que não precisam estar sempre alinhadas. Mais tarde, com percepção, deve ganhar autoridade
para manobras que o modelo linear não cobre, como degraus maiores.

```text
 comandos (velocidade, giro, altura, pousar)        sensores do robô (IMU, encoders, torques)
            │                                                   │
            ▼                                                   ▼
 ┌─ RL residual (50 Hz) ─────────────────────────────────────────────────────┐   fase atual: cego
 │  Δcomprimento e Δângulo de cada perna, Δtorque de cada roda                │   (proprioceptivo)
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    ▼
 ┌─ model-based (1 kHz) ─────────────────────────────────────────────────────┐
 │  modo automático (estacionado / levantando / operando / pousando)          │
 │  LQR K(L0) → torque da roda T e torque virtual da perna Tp                 │
 │  PIDs: comprimento, roll, yaw, sincronismo das pernas                      │
 │  VMC: jacobiano do 5 barras fechado → torques dos 4 quadris                │
 └──────────────────────────────────┬────────────────────────────────────────┘
                                    ▼
                         6 motores 9:1 + 2 servos do trem
```

---

## Parte 1 — Model-based

### Onde estamos

| bloco | estado |
|---|---|
| modelo | MuJoCo gerado do CAD: massas, inércias e pernas 5 barras fechadas por restrições de laço (folga < 0,1 mm) |
| LQR | modelo planar de meio robô (roda + perna de comprimento variável + corpo), linearizado em cada L0 com a geometria real do 5 barras. Ganhos ajustados por cúbicos em L0, **de 0,17 a 0,54 m** (erro do ajuste < 0,3%) |
| VMC | força ao longo da perna + torque virtual → torques dos quadris por τ = Jᵀ·F, com o jacobiano do laço fechado (analítico em `control/five_bar.py`) |
| PIDs em cascata | comprimento (com peso por perna), roll, yaw (taxa + correção de heading), sincronismo das pernas |
| estado | só o que o robô real mede: IMU, odometria pelos encoders corrigida pelo giro da canela, contato estimado pelos torques dos quadris |
| faixa da perna | **0,18 a 0,52 m**. Medido com as malhas de colisão: a roda bate no chassi abaixo de ~0,17 m; a perna toda esticada fica em 0,565 m |
| trem de pouso | **automático**: nasce estacionado (0,18 m, trem abaixado), levanta, recolhe o trem quando está parado, opera em qualquer altura e pousa quando pedido. O usuário só pede para operar ou pousar |

Validado no MuJoCo (CPU):
- parado com pitch estável em +1,7°;
- velocidades de −1 a 1,5 m/s e giro de até 1,5 rad/s;
- empurrão de 80 N recuperado em ~1 s;
- arena com rampa de 13°;
- altura de 0,18 a 0,52 m, parado e andando;
- ciclo completo de pouso (levanta em 1,9 s, pousa em 3,3 s);
- IMU Mahony com ruído.

O port para GPU (`sim/warp_rl/controller.py`) bate com o controlador numpy: rodas a 0,45%, quadris a 2%.

### Decisões e o porquê

- **LQR com ganhos em função de L0, a partir da geometria real.** O centro de massa da perna fica perto da roda, porque o motor da roda está na canela. Usar "CoM no meio da perna" dava ganhos errados.
- **A faixa da perna vem da geometria, não do ajuste.** O antigo limite de 0,36 m era só a faixa em que os ganhos tinham sido ajustados.
- **Estado só com sensores reais.** O mesmo código roda no robô. O estado perfeito do simulador fica só para comparação e para o *critic* do RL.
- **A condição "parado" para recolher o trem não usa o pitch absoluto.** O pitch de equilíbrio depende do CoM, que no robô real é incerto.

### Próximos passos

1. **Caracterizar os atuadores:** torque de pico e contínuo, inércia do rotor, folga, constante de torque. Depois, regenerar o MJCF e o LQR (`generate_mjcf.py`, `lqr_design.py --write`). Hoje usamos 18 N·m na junta, e o real deve ser maior.
2. **Atualizar a massa do robô montado,** que é um pouco menor que os 15,7 kg do CAD.
3. **Portar para o robô** (`robot/`): drivers moteus, IMU, o mesmo estimador do modo `robot` e o laço a 1 kHz. Comparar logs reais com a simulação.
4. **Ajustar Q/R para as alturas grandes,** de 0,45 a 0,52 m, onde a perna é mais lenta e o corpo fica mais alto.
5. **Melhorar o estimador de contato e testar a suspensão ativa,** cujos ganhos hoje estão em zero, para terreno irregular.

---

## Parte 2 — Aprendizado por reforço

### De onde viemos

O primeiro treino, no Isaac, deu ao PPO o **torque bruto dos 6 motores**. A política não aprendeu a
equilibrar: ela abaixou o trem de pouso e ficou sentada nele. Isso aconteceu porque o alvo do servo nunca
foi definido, a recompensa pagava quase o máximo para o robô torto, a terminação era frouxa e o
curriculum estava mal feito. Detalhes em [terrain-run-2026-10-31.md](terrain-run-2026-10-31.md).

**Lição:** com um LQR que já funciona, aprender tudo do zero desperdiça amostras, é frágil para o
sim-to-real (os atuadores não foram medidos) e convida a política a trapacear.

### Abordagem atual: RL residual em GPU no MuJoCo Warp

Código em [`sim/warp_rl/`](../sim/warp_rl/README.md).

- **Ação, por perna, só no modo `operating`:**
  - Δcomprimento de cada perna: a média muda a altura, a diferença inclina o corpo em roll;
  - Δângulo de cada perna: a média desloca o equilíbrio, a diferença abre uma perna à frente da outra;
  - Δtorque em cada roda.

  A última camada do actor começa zerada. Ou seja, o treino começa exatamente do model-based.
- **Física:** MuJoCo Warp com o mesmo modelo do CAD em que o LQR foi validado. São 4096 robôs, com física e controlador a 1 kHz, todos num grafo CUDA (~45 mil passos de política por segundo numa RTX 5070).
- **Terreno grande compartilhado,** como no Isaac: 6 tipos (plano, irregular, rampa até 15°, ondas, degraus até 5 cm, blocos) × 6 níveis, com curriculum por nível.
- **Actor:** só o que o robô mede (IMU, comprimento e ângulo das pernas, velocidade das rodas, odometria), mais os comandos e 5 quadros de histórico, com ruído. **Critic:** também recebe a velocidade real, um mapa de alturas e os parâmetros sorteados.
- **Robustez:** atrito de 0,5 a 1,25, massa do chassi ±15%, CoM ±2 cm, empurrões de até 0,6 m/s, 15% dos episódios começando estacionados.
- **Recompensa:** seguir velocidade, giro e altura; penalidades para inclinação, sacudida, tamanho e variação do residual, torque, trem tocando o chão operando, e queda.

**Resultado (v2, 1500 iterações, ~147 milhões de passos).** A avaliação usou 2160 robôs por 60 s, num terreno gerado com outra semente. As duas colunas rodaram nos mesmos cenários.

| | só model-based | model-based + RL |
|---|---:|---:|
| quedas por minuto de robô | 0,265 | **0,069** |
| erro de velocidade (m/s) | 0,207 | **0,094** |

O RL melhora em todos os 18 grupos de terreno. A maior diferença está nos degraus difíceis (de 0,89 para 0,18 quedas/min). O ponto mais fraco é a rampa de 15° (0,25 quedas/min). Detalhes em `experiments/2026-10-31_20-52-28-warp-residual-v2/evaluation.md` (fora do Git).

Lição de treino: num residual, o ruído de exploração age direto no robô. Ele precisa ficar limitado
(desvio entre 0,05 e 0,3, sem bônus de entropia). Na primeira tentativa ele cresceu até 0,66 e as quedas
foram de 2% para 75%.

### Limitações atuais

- Para o treino ser rápido, o heightfield tem 8 cm de resolução, e a quina de um degrau vira uma rampa de 8 cm.
- Só pneus e pontas do trem colidem com o terreno.
- O residual não age durante levantar e pousar.
- Torque e massa são provisórios.
- Ainda não houve teste fora do ambiente de treino (MuJoCo de CPU, robô real).

### Roteiro

1. **Consolidar o residual cego**
   - Rodar a política no MuJoCo de CPU, com o viewer e o controle do Xbox.
   - Exportar o actor (TorchScript) para rodar a 50 Hz junto com o laço de 1 kHz.
2. **Preparar o sim-to-real**
   - Latência de sensor e de atuador, modelo de atuador medido, ruído e bias de IMU realistas.
   - Randomizar torque máximo e folga.
   - Depois, testar no robô com a autoridade do residual reduzida e aumentá-la aos poucos.
3. **Mais autoridade**
   - Escalas maiores do residual por curriculum.
   - Depois, uma política com alvos absolutos de comprimento e ângulo por perna, destilada do model-based + residual, para manobras fora do modelo linear (como um impulso das duas rodas juntas).
4. **Percepção e degraus grandes (Isaac)**
   - Câmera de profundidade → nuvem de pontos → mapa de elevação em volta do robô → amostras de altura na política.
   - Treino professor-aluno: o professor tem o mapa perfeito, o aluno tem o mapa ruidoso e o histórico.
   - Levar as lições do post-mortem do Isaac: alvo do trem, atrito, roda redonda, terminação e curriculum.
   - **Limite físico:** sem abdução de quadril, o robô não tem como andar uma perna de cada vez por muito tempo. Degraus maiores que o raio da roda (~10 cm) devem ser vencidos com as duas rodas juntas, num impulso de encolher e esticar. O curso da perna (0,18 a 0,52 m) ajuda nisso.

**Regra:** toda política nova é avaliada contra o model-based sozinho, nos mesmos cenários (`eval.py`).
Uma política só substitui o model-based onde for melhor que ele.
