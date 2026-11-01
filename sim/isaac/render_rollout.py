"""Software-render recorded Isaac poses or preview the course; never resimulates a policy.

The replay uses MuJoCo only as a viewer. Isaac physics and policy determine the
recorded motion. Preview mode is an orbit of the initial pose, not a trained run.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',default=shutil.which('ffmpeg') or str(Path.home()/'anaconda3/envs/audio/bin/ffmpeg'))
    p.add_argument('--width',type=int,default=960)
    p.add_argument('--height',type=int,default=540)
    args=p.parse_args()
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('EGL_PLATFORM','surfaceless')
    os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES','/usr/share/glvnd/egl_vendor.d/50_mesa.json')
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    from primus_task.terrain import course_mesh
    from kinematics import nominal_pose
    repo=Path(__file__).resolve().parents[2]
    scene=ET.parse(repo/'sim/models/cena_plana.xml').getroot()
    scene.remove(scene.find('include'))
    robot=ET.parse(repo/'sim/models/robo_bipedal.xml').getroot()
    for mesh in robot.findall('.//mesh'):
        mesh.set('file',str(repo/'sim/models/meshes'/mesh.get('file')))
    for child in robot: scene.append(child)
    asset=scene.find('asset');world=scene.find('worldbody')
    # Multiple asset/worldbody blocks are legal MJCF sections and merge on compile.
    vertices,faces=course_mesh()
    # MJCF mesh rendering preserves triangles; collision is unused in replay.
    ET.SubElement(asset,'mesh',name='course',vertex=' '.join(map(str,vertices.ravel())),
                  face=' '.join(map(str,faces.ravel())))
    world.remove(world.find("geom[@name='ground']"))
    ET.SubElement(world,'geom',type='mesh',mesh='course',rgba='0.35 0.45 0.38 1',contype='0',conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(scene,encoding='unicode'))
    data=mujoco.MjData(model);renderer=mujoco.Renderer(model,args.height,args.width)
    options=mujoco.MjvOption();options.geomgroup[3]=0
    cam=mujoco.MjvCamera();cam.elevation=-22
    trajectory=np.load(args.trajectory) if args.trajectory else None
    if trajectory is not None:
        poses=trajectory['poses'];angles=trajectory['joints'];names=trajectory['joint_names']
        fps=round(1/float(trajectory['dt']))
        title='Isaac '+str(trajectory['device'])+' policy rollout | MuJoCo pose replay'
    else:
        q,z=nominal_pose();poses=np.tile([0,-0.6,z,0,0,0,1],(250,1))
        names=np.array(list(q));angles=np.tile(list(q.values()),(250,1));fps=25
        title='Environment preview | initial pose | NO TRAINED POLICY'
    joint_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,str(n)) for n in names]
    if min(joint_ids)<0: raise RuntimeError('Recorded joint missing from rendering model')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    command=[args.ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
             '-s',f'{args.width}x{args.height}','-r',str(fps),'-i','-',
             '-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(args.output)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index,pose in enumerate(poses):
            data.qpos[:3]=pose[:3];data.qpos[3:7]=pose[[6,3,4,5]]
            for jid,angle in zip(joint_ids,angles[index]):data.qpos[model.jnt_qposadr[jid]]=angle
            mujoco.mj_forward(model,data)
            if trajectory is None:
                cam.lookat[:]=[0,4,0];cam.distance=10;cam.azimuth=155+30*np.sin(index/80)
            else:
                cam.lookat[:]=pose[:3]+[0,0.5,-0.1];cam.distance=2.5;cam.azimuth=155
            renderer.update_scene(data,cam,scene_option=options)
            frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,args.width,52),fill=(15,20,25))
            draw.text((14,10),title,fill='white')
            if trajectory is None: caption='Flat | 8% ramp | 2.5 cm stairs | uneven patch'
            else: caption=f"Section: {['flat','ramp','stairs','uneven'][int(trajectory['stages'][index])]} | recorded frame {index}"
            draw.text((14,30),caption,fill='white')
            encoder.stdin.write(np.asarray(frame).tobytes())
    finally:
        renderer.close();encoder.stdin.close();result=encoder.wait()
    if result: raise SystemExit('Video encoder failed: '+str(result))
    print(args.output)


if __name__=='__main__': main()
