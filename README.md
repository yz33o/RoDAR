git clone https://github.com/flow-project/flow.git

tensorboard --logdir=Comparative_experiment/Adversial_mltitrpo/logs --port=6006

Change the IDM model in the flow to the following format：
class IDMController(BaseController):
    def __init__(self,
                 veh_id,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        BaseController.__init__(
            self,
            veh_id,
            car_following_params,
            delay=time_delay,
            fail_safe=fail_safe,
            noise=noise,
            display_warnings=display_warnings,
        )
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0

    def get_accel(self, env):
        v = env.k.vehicle.get_speed(self.veh_id)
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        h = env.k.vehicle.get_headway(self.veh_id)

        if abs(h) < 1e-3:
            h = 1e-3

        if lead_id is None or lead_id == '':
            s_star = 0
        else:
            lead_vel = env.k.vehicle.get_speed(lead_id)
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                   (2 * np.sqrt(self.a * self.b))
            )
        # ======== 根据需要选择模式 ========
        # ======== IDM 原始加速度 ========
        # acc = self.a * (1 - (v / self.v0) ** self.delta - (s_star / h) ** 2)
        # ======== 训练使用    加扰动项 ========
        # sigma = 0.2
        # m_i = np.random.normal(loc=1.0, scale=sigma)
        # m_i = np.clip(m_i, 0.0, 2.0)
        # acc = acc + self.delta_adv * m_i
        # ======== 测试使用    加随机扰动项 ========
          acc = acc + np.random.uniform(-2.2, 0.5)
        return acc

Demonstration video:

<table>
  <!-- ====== 第一行 ====== -->
  <tr> 
    <td width="50%">
      <b>训练前:</b><br>
      <video src="https://github.com/user-attachments/assets/a4ea1005-1787-4a49-8c52-9a720f5b6b12" controls="controls" width="100%"></video>
    </td>
    <td width="50%">
      <b>训练后:</b><br>
      <video src="https://github.com/user-attachments/assets/690f50b3-c2db-417d-8e3c-032050d3fb7a" controls="controls" width="100%"></video>
    </td>
  </tr>
  
  <!-- ====== 第二行 ====== -->
  <tr> 
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/51791b90-0f90-45e5-9881-e052754c88ec" controls="controls" width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/f5d56e3f-f76a-493b-9772-b982cba2071e" controls="controls" width="100%"></video>
    </td>
  </tr>
  <!-- ====== 第三行 ====== -->
  <tr> 
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/d797f958-5c28-4858-8e50-c4bd4d9562c2" controls="controls" width="100%"></video>
    </td>
    <td width="50%">
      <video src="https://github.com/user-attachments/assets/bfabe974-db89-4c5c-ae15-bcdf69b709cf" controls="controls" width="100%"></video>
    </td>
  </tr>
</table>



















