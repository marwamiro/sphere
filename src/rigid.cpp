// rigid.cpp -- Rigid body solver using the Bullet Physics engine
#include <iostream>
#include <string>
#include <cstdio>

// Bullet is installed using the Debian packages.
// Documentation is located in /usr/share/doc/libbullet*
#include "btBulletDynamicsCommon.h"

#include "sphere.h"
#include "datatypes.h"
#include "utility.cuh"
#include "utility.h"
#include "constants.cuh"
#include "debug.h"



void DEM::startTimeRigid() {

    //// WORLD INITIALIZATION
    
    // collision configuration contains default setup for memory, collision
    // setup. Advanced users can create their own configuration.
    btDefaultCollisionConfiguration* collisionConfiguration
        = new btDefaultCollisionConfiguration();

    // use the default collision dispatcher. For parallel processing you can use
    // a different dispatcher (see Extras/BulletMultiThreaded)
    btCollisionDispatcher* dispatcher
        = new btCollisionDispatcher(collisionConfiguration);

    // btDbvtBroadphase is a good general purpose broadphase. You can also try
    // out btAxis3Sweep.
    btBroadphaseInterface* overlappingPairCache
        = new btDbvtBroadphase();

    // the default constraint solver. For parallel processing you can use a
    // different solver (see also Extras/BulletMultiThreaded)
    btSequentialImpulseConstraintSolver* solver
        = new btSequentialImpulseConstraintSolver;

    btDiscreteDynamicsWorld* dynamicsWorld
        = new btDiscreteDynamicsWorld(
                dispatcher, overlappingPairCache,
                solver, collisionConfiguration);

    dynamicsWorld->setGravity(
            btVector3(params.g[0], params.g[1], params.g[2]));


    //// OBJECT INITIALIZATION

    // keep track of the shapes, memory is released at exit.
    // reuse collision shapes among rigid bodies whenever possible.
    btAlignedObjectArray<btCollisionShape*> collisionShapes;

    double wallthickness = 1.0;

    Float restitution      = params.k_n;
    Float friction         = params.mu_s;
    Float rollingFriction  = params.mu_r;
    Float linearDamping    = params.gamma_n;
    Float angularDamping   = params.gamma_r;

    // create static floor (-z)
    {
        Float mass = 0.0;
        btVector3 size(grid.L[0], grid.L[1], wallthickness);
        btVector3 origin(0.0, 0.0, -wallthickness);
        btVector3 localInertia(0.0, 0.0, 0.0);
        addRigidBox(
                dynamicsWorld,
                collisionShapes,
                mass,
                size,
                origin,
                localInertia,
                restitution,
                friction,
                rollingFriction,
                linearDamping,
                angularDamping);
    }

    // create static -x boundary
    {
        Float mass = 0.0;
        btVector3 size(wallthickness, grid.L[1], grid.L[2]);
        btVector3 origin(-wallthickness, 0.0, 0.0);
        btVector3 localInertia(0.0, 0.0, 0.0);
        addRigidBox(
                dynamicsWorld,
                collisionShapes,
                mass,
                size,
                origin,
                localInertia,
                restitution,
                friction,
                rollingFriction,
                linearDamping,
                angularDamping);
    }

    // create static +x boundary
    {
        Float mass = 0.0;
        btVector3 size(wallthickness, grid.L[1], grid.L[2]);
        btVector3 origin(grid.L[0], 0.0, 0.0);
        btVector3 localInertia(0.0, 0.0, 0.0);
        addRigidBox(
                dynamicsWorld,
                collisionShapes,
                mass,
                size,
                origin,
                localInertia,
                restitution,
                friction,
                rollingFriction,
                linearDamping,
                angularDamping);
    }

    // create static -y boundary
    {
        Float mass = 0.0;
        btVector3 size(grid.L[0], wallthickness, grid.L[2]);
        btVector3 origin(0.0, -wallthickness, 0.0);
        btVector3 localInertia(0.0, 0.0, 0.0);
        addRigidBox(
                dynamicsWorld,
                collisionShapes,
                mass,
                size,
                origin,
                localInertia,
                restitution,
                friction,
                rollingFriction,
                linearDamping,
                angularDamping);
    }

    // create static +y boundary
    {
        Float mass = 0.0;
        btVector3 size(grid.L[0], wallthickness, grid.L[2]);
        btVector3 origin(0.0, grid.L[1], 0.0);
        btVector3 localInertia(0.0, 0.0, 0.0);
        addRigidBox(
                dynamicsWorld,
                collisionShapes,
                mass,
                size,
                origin,
                localInertia,
                restitution,
                friction,
                rollingFriction,
                linearDamping,
                angularDamping);
    }


    // create dynamic objects
    {
        Float mass, radius;
        btVector3 origin, localInertia;

        for (int i=0; i<np; ++i) {

            radius = k.x[i].w;
            mass = 4.0/3.0 * radius*radius*radius * params.rho;
            origin = btVector3(k.x[i].x, k.x[i].y, k.x[i].z);
            localInertia = btVector3(0.0, 0.0, 0.0);
            addRigidSphere(
                    dynamicsWorld,
                    collisionShapes,
                    mass,
                    radius,
                    origin,
                    localInertia,
                    restitution,
                    friction,
                    rollingFriction,
                    linearDamping,
                    angularDamping);
        }
    }


    //// CLEANUP

    // remove the rigidbodies from the dynamics world and delete them
    for (int i=dynamicsWorld->getNumCollisionObjects()-1; i>=0; i--) {

        btCollisionObject* obj
            = dynamicsWorld->getCollisionObjectArray()[i];

        btRigidBody* body
            = btRigidBody::upcast(obj);

        if (body && body->getMotionState()) {
            delete body->getMotionState();
        }
        dynamicsWorld->removeCollisionObject(obj);
        delete obj;
    }

    // delete collision shapes
    for (int j=0; j<collisionShapes.size(); ++j) {
        btCollisionShape* shape
            = collisionShapes[j];
        collisionShapes[j] = 0;
        delete shape;
    }

    delete dynamicsWorld;
    delete solver;
    delete overlappingPairCache;
    delete dispatcher;
    delete collisionConfiguration;

    collisionShapes.clear();

}

// Add box as rigid body with collision shape
void DEM::addRigidBox(
        btDiscreteDynamicsWorld* dynamicsWorld,
        btAlignedObjectArray<btCollisionShape*> &collisionShapes,
        Float mass,
        btVector3 &size,
        btVector3 &origin,
        btVector3 &localInertia,
        Float restitution,
        Float friction,
        Float rollingFriction,
        Float linearDamping,
        Float angularDamping)
{
    btCollisionShape* boxShape = new btBoxShape(size);
    collisionShapes.push_back(boxShape);

    btTransform boxTransform;
    boxTransform.setIdentity();
    boxTransform.setOrigin(origin);

    // a rigidbody is dynamic if and only if mass is nonzero, otherwise
    // static
    bool isDynamic = (mass != 0.f);
    if (isDynamic)
        boxShape->calculateLocalInertia(mass, localInertia);

    // using motionstate is recommended. It provides interpolation
    // capabilities, and only synchronizes 'active' objects
    btDefaultMotionState* myMotionState
        = new btDefaultMotionState(boxTransform);

    btRigidBody::btRigidBodyConstructionInfo 
        rbInfo(mass, myMotionState, boxShape, localInertia);
    rbInfo.m_restitution     = btScalar(restitution);
    rbInfo.m_friction        = btScalar(friction);
    rbInfo.m_rollingFriction = btScalar(rollingFriction);
    rbInfo.m_linearDamping   = btScalar(linearDamping);
    rbInfo.m_angularDamping  = btScalar(angularDamping);

    // add the body to the dynamics world
    btRigidBody* body = new btRigidBody(rbInfo);
    dynamicsWorld->addRigidBody(body);
}

// Add sphere as rigid body with collision shape
void DEM::addRigidSphere(
        btDiscreteDynamicsWorld* dynamicsWorld,
        btAlignedObjectArray<btCollisionShape*> &collisionShapes,
        Float mass,
        Float radius,
        btVector3 &origin,
        btVector3 &localInertia,
        Float restitution,
        Float friction,
        Float rollingFriction,
        Float linearDamping,
        Float angularDamping)
{
    btCollisionShape* sphereShape = new btSphereShape(radius);
    collisionShapes.push_back(sphereShape);

    btTransform sphereTransform;
    sphereTransform.setIdentity();
    sphereTransform.setOrigin(origin);

    // a rigidbody is dynamic if and only if mass is nonzero, otherwise
    // static
    bool isDynamic = (mass != 0.f);
    if (isDynamic)
        sphereShape->calculateLocalInertia(mass, localInertia);

    // using motionstate is recommended. It provides interpolation
    // capabilities, and only synchronizes 'active' objects
    btDefaultMotionState* myMotionState
        = new btDefaultMotionState(sphereTransform);

    btRigidBody::btRigidBodyConstructionInfo 
        rbInfo(mass, myMotionState, sphereShape, localInertia);
    rbInfo.m_restitution     = btScalar(restitution);
    rbInfo.m_friction        = btScalar(friction);
    rbInfo.m_rollingFriction = btScalar(rollingFriction);
    rbInfo.m_linearDamping   = btScalar(linearDamping);
    rbInfo.m_angularDamping  = btScalar(angularDamping);

    // add the body to the dynamics world
    btRigidBody* body = new btRigidBody(rbInfo);
    dynamicsWorld->addRigidBody(body);
}

