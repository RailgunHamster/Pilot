#include "runtime/function/controller/character_controller.h"

#include "runtime/core/base/macro.h"

#include "runtime/function/framework/component/motor/motor_component.h"
#include "runtime/function/framework/world/world_manager.h"
#include "runtime/function/global/global_context.h"
#include "runtime/function/physics/physics_scene.h"
#include <optional>

namespace Pilot
{
    CharacterController::CharacterController(const Capsule& capsule) : m_capsule(capsule)
    {
        m_rigidbody_shape                                    = RigidBodyShape();
        m_rigidbody_shape.m_geometry                         = PILOT_REFLECTION_NEW(Capsule);
        *static_cast<Capsule*>(m_rigidbody_shape.m_geometry) = m_capsule;

        m_rigidbody_shape.m_type = RigidBodyShapeType::capsule;

        Quaternion orientation;
        orientation.fromAngleAxis(Radian(Degree(90.f)), Vector3::UNIT_X);

        m_rigidbody_shape.m_local_transform =
            Transform(Vector3(0, 0, capsule.m_half_height + capsule.m_radius), orientation, Vector3::UNIT_SCALE);
    }

    Vector3 CharacterController::move(const Vector3& current_position, const Vector3& displacement)
    {
        std::shared_ptr<PhysicsScene> physics_scene =
            g_runtime_global_context.m_world_manager->getCurrentActivePhysicsScene().lock();
        ASSERT(physics_scene);

        std::vector<PhysicsHitInfo> hits;

        Transform world_transform =
            Transform(current_position + 0.1f * Vector3::UNIT_Z, Quaternion::IDENTITY, Vector3::UNIT_SCALE);

        Vector3 vertical_displacement   = displacement.z * Vector3::UNIT_Z;
        Vector3 horizontal_displacement = Vector3(displacement.x, displacement.y, 0.f);

        Vector3 vertical_direction   = vertical_displacement.normalisedCopy();
        Vector3 horizontal_direction = horizontal_displacement.normalisedCopy();

        Vector3 final_position = current_position;

        m_is_touch_ground = physics_scene->sweep(
            m_rigidbody_shape, world_transform.getMatrix(), Vector3::NEGATIVE_UNIT_Z, 0.105f, hits);

        hits.clear();

        // side pass
        bool side_zero = false;
        if (physics_scene->sweep(m_rigidbody_shape,
                                 world_transform.getMatrix(),
                                 horizontal_direction,
                                 horizontal_displacement.length(),
                                 hits))
        {
            const float distance = hits[0].hit_distance;
            side_zero            = distance <= .001f;
            if (side_zero)
            {
                const float   length = 1.f / (horizontal_direction.dotProduct(-hits[0].hit_normal));
                const Vector3 v      = (hits[0].hit_normal + horizontal_direction * length).normalisedCopy();
                final_position += v * horizontal_displacement.length() * v.dotProduct(horizontal_direction);
            }
            else
            {
                final_position += distance * horizontal_direction;
            }
        }
        else
        {
            final_position += horizontal_displacement;
        }

        hits.clear();

        world_transform.m_position -= 0.1f * Vector3::UNIT_Z;

        // vertical pass
        if (physics_scene->sweep(m_rigidbody_shape,
                                 world_transform.getMatrix(),
                                 vertical_direction,
                                 vertical_displacement.length(),
                                 hits))
        {
            if (side_zero && hits[0].hit_distance < .001f)
            {
                if (hits.size() > 1)
                {
                    final_position += hits.back().hit_distance * vertical_direction;
                }
                else if (vertical_direction.z > 0.f)
                {
                    final_position += vertical_displacement;
                }
            }
            else
            {
                final_position += hits[0].hit_distance * vertical_direction;
            }
        }
        else
        {
            final_position += vertical_displacement;
        }

        return final_position;
    }

} // namespace Pilot
