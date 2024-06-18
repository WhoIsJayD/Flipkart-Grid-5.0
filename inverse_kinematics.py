# inverse_kinematics.py
import math


class InverseKinematics:
    import math

    # Defining Link Lengths
    d1 = 9.5
    d2 = 55
    d3 = 52
    d4 = 5.5
    d5 = 26  # +3.5

    # Defining Rotation Matrix
    nx = 1
    ny = 0
    nz = 0

    sx = 0
    sy = -1
    sz = 0

    ax = 0
    ay = 0
    az = -1

    @staticmethod
    def angle_constraint(angle):
        if 0 > angle > -90:
            angle = angle + 90
        if 0 > angle < -90:
            angle = abs(angle) + 90
        if 90 < angle < 180:
            angle = angle - 90
        if 90 < angle > 180:
            angle = angle - 180  # or 270 - angle1
        return angle

    # @staticmethod
    # def linear_convert(value, input_min, input_max, output_min, output_max):
    #     demap_angle = (value - input_min) / (input_max - input_min) * (output_max - output_min) + output_min
    #     return demap_angle

    @staticmethod
    def calculate_inverse_kinematics(x, y, z):
        q1 = math.atan2(y, x)

        eq1 = -(ax * math.cos(q1) + ay * math.sin(q1))
        eq2 = -az

        q5 = math.atan2((nx * math.sin(q1) - ny * math.cos(q1)), (sx * math.sin(q1) - sy * math.cos(q1)))

        c = (x / math.cos(q1)) + (d5 * eq1) - (d4 * eq2)
        d = (d1 - (d4 * eq1) - (d5 * eq2) - z)

        R = (c * c + d * d - d3 * d3 - d2 * d2) / (2 * (d3 * d2))
        t = math.sqrt(1 - R * R)

        q3 = math.atan2(t, R)

        r = d3 * math.cos(q3) + d2
        s = d3 * math.sin(q3)

        q2 = math.atan2((r * d) - (s * c), (r * c) + (s * d))

        eq3 = math.atan2(-(ax * math.cos(q1) + ay * math.sin(q1)), -az)
        q4 = eq3 - (q2 + q3)

        # Convert angles to degrees
        angle1 = math.degrees(q1)
        angle2 = math.degrees(q2)
        angle3 = math.degrees(q3)
        angle4 = math.degrees(q4)
        angle5 = math.degrees(q5)

        # Apply angle constraints
        angle1 = InverseKinematics.angle_constraint(angle1)
        angle2 = InverseKinematics.angle_constraint(angle2)
        angle3 = InverseKinematics.angle_constraint(angle3)
        angle4 = InverseKinematics.angle_constraint(angle4)
        angle5 = InverseKinematics.angle_constraint(angle5)

        # Apply demap for the 2nd motor
        # angle2 = InverseKinematics.linear_convert(angle2, 0, 360, 0, 8192)

        return angle1, angle2, angle3, angle4, angle5


inverse = InverseKinematics.calculate_inverse_kinematics(0, 62, 15)
print(inverse)
