# DICE Dial references and reading list

This is the technical reading list behind DICE Dial. It separates work that directly informed the implementation from broader references that are useful for understanding dexterous manipulation, asymmetric reinforcement learning, dynamics randomization, and the simulation stack.

DICE Dial is a simulation project. The references on domain randomization and sim-to-real transfer motivate future extensions; they are **not** evidence that this repository itself demonstrates real-robot transfer.

## Dexterous in-hand manipulation

1. **Andrychowicz, M. et al. / OpenAI (2020). _Learning Dexterous In-Hand Manipulation_. International Journal of Robotics Research.**
   [arXiv](https://arxiv.org/abs/1808.00177)
   Directly relevant Shadow Dexterous Hand work: simulation-trained reinforcement learning for object reorientation, multi-finger coordination, finger gaiting, and domain randomization for real-robot transfer. This is the closest historical reference to DICE Dial's underlying manipulation setting.

2. **Akkaya, I. et al. / OpenAI (2019). _Solving Rubik's Cube with a Robot Hand_.**
   [arXiv](https://arxiv.org/abs/1910.07113)
   A key reference for long-horizon sequential in-hand manipulation with a Shadow Hand and for automatic domain randomization. DICE Dial is much narrower and remains simulation-only, but the paper is an important benchmark for the broader research direction.

3. **Rajeswaran, A. et al. (2018). _Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations_. Robotics: Science and Systems.**
   [arXiv](https://arxiv.org/abs/1709.10087)
   Early evidence that model-free deep RL can scale to high-dimensional, contact-rich dexterous-hand control. DICE Dial does not use demonstrations, but the manipulation and control setting is closely related.

4. **Zhu, H. et al. (2019). _Dexterous Manipulation with Deep Reinforcement Learning: Efficient, General, and Low-Cost_. ICRA.**
   [arXiv](https://arxiv.org/abs/1810.06045)
   Useful background on direct deep-RL approaches to contact-rich multi-finger manipulation and the practical difficulties of learning dexterous control.

## Reinforcement learning and asymmetric training

5. **Schulman, J. et al. (2017). _Proximal Policy Optimization Algorithms_.**
   [arXiv](https://arxiv.org/abs/1707.06347)
   The PPO algorithm used by the final policy through RSL-RL.

6. **Schulman, J. et al. (2015). _High-Dimensional Continuous Control Using Generalized Advantage Estimation_.**
   [arXiv](https://arxiv.org/abs/1506.02438)
   Defines generalized advantage estimation (GAE), used here with `gamma = 0.99` and `lambda = 0.95`.

7. **Pinto, L. et al. (2017). _Asymmetric Actor Critic for Image-Based Robot Learning_.**
   [arXiv](https://arxiv.org/abs/1710.06542)
   Core reference for using privileged simulator state in the critic while constraining the actor to a smaller observation interface. DICE Dial follows this asymmetric-training idea, although its actor is state-based rather than image-based.

8. **Ng, A. Y., Harada, D., and Russell, S. (1999). _Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping_. ICML.**
   [paper](https://luthuli.cs.uiuc.edu/~daf/courses/Games/AIpapers/ml99-shaping.pdf)
   Foundational reward-shaping reference. DICE Dial uses differential angular progress and signed hold-progress shaping, but does **not** claim that its complete reward is an exact policy-invariant potential-based transformation.

## Geometry and state representation

9. **Zhou, Y. et al. (2019). _On the Continuity of Rotation Representations in Neural Networks_. CVPR.**
   [arXiv](https://arxiv.org/abs/1812.07035)
   Motivates the continuous 6D rotation representation used in the actor instead of feeding a quaternion directly.

## Dynamics randomization and sim-to-real background

10. **Tobin, J. et al. (2017). _Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World_. IROS.**
    [arXiv](https://arxiv.org/abs/1703.06907)
    Seminal domain-randomization paper. Its focus is visual transfer, but the central idea of exposing a model to simulator variation is foundational to later sim-to-real work.

11. **Peng, X. B. et al. (2018). _Sim-to-Real Transfer of Robotic Control with Dynamics Randomization_. ICRA.**
    [arXiv](https://arxiv.org/abs/1710.06537)
    Particularly relevant to DICE Dial's future work because it randomizes physical dynamics during training. The final DICE Dial policy intentionally did **not** use training-time mass/friction randomization; its ±20% material evaluation is held out.

12. **Akkaya, I. et al. (2019). _Solving Rubik's Cube with a Robot Hand_.**
    [arXiv](https://arxiv.org/abs/1910.07113)
    Also belongs in this section for its Automatic Domain Randomization method and its demonstration that robust sequential dexterous policies can support sim-to-real transfer when paired with substantially broader randomization and a real-robot system.

## Simulation and learning software

13. **Mittal, M. et al. (2025). _Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning_.**
    [arXiv](https://arxiv.org/abs/2511.04831) · [Isaac Lab repository](https://github.com/isaac-sim/IsaacLab)
    DICE Dial is implemented as a custom Isaac Lab Direct environment and inherits the Shadow Hand articulation, stock DexCube, contact simulation, reset machinery, and low-level action path from the framework.

14. **Mittal, M. et al. (2023). _Orbit: A Unified Simulation Framework for Interactive Robot Learning Environments_. IEEE Robotics and Automation Letters.**
    [arXiv](https://arxiv.org/abs/2301.04195)
    The predecessor framework whose modular GPU-parallel robot-learning design developed into Isaac Lab.

15. **Schwarke, C., Mittal, M., Rudin, N., Hoeller, D., and Hutter, M. (2025). _RSL-RL: A Learning Library for Robotics Research_.**
    [arXiv](https://arxiv.org/abs/2509.10771) · [RSL-RL repository](https://github.com/leggedrobotics/rsl_rl)
    GPU-oriented robot-learning library used for PPO collection, optimization, checkpointing, and actor/critic execution.

16. **NVIDIA Isaac Sim 5.1 documentation.**
    [release notes](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html) · [physics fundamentals](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/physics/simulation_fundamentals.html)
    Version-specific documentation for the simulator runtime used by the final experiment.

17. **NVIDIA PhysX documentation — rigid-body dynamics and friction.**
    [rigid-body dynamics](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/RigidBodyDynamics.html) · [Omni Physics material combine modes](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html)
    Relevant to interpreting object mass, static/dynamic friction, and the material-combination caveat in the adverse stress test.

## Suggested reading order

For the shortest path through the literature:

1. Andrychowicz et al. — understand the Shadow Hand in-hand manipulation problem.
2. Akkaya et al. — see the long-horizon Rubik's Cube extension and Automatic Domain Randomization.
3. Schulman et al. (PPO) + Pinto et al. (asymmetric actor-critic) — understand the learning architecture used here.
4. Zhou et al. — understand the 6D rotation representation.
5. Peng et al. — understand the most natural next step for training-time dynamics robustness.
6. Isaac Lab + RSL-RL papers — understand the software stack and large-scale GPU-parallel workflow.
