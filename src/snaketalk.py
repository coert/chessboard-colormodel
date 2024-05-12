import asyncio
import cv2
import numpy as np
from time import time

CHESS_WIDTH = 13  # Number of squares in the width of the chessboard
CHESS_HEIGHT = 6  # Number of squares in the height of the chessboard

NB_CALIB_CHECKS = 50  # Number of calibration checks before calibration
THETA = 0.45  # Probability threshold for the color model


def draw_a_chess_board():
    try:
        chessboard_image = np.zeros((CHESS_HEIGHT * 100, CHESS_WIDTH * 100, 3), np.uint8)
        for i in range(CHESS_HEIGHT):
            for j in range(CHESS_WIDTH):
                if (i + j) % 2 == 0:
                    chessboard_image[i * 100 : (i + 1) * 100, j * 100 : (j + 1) * 100] = [
                        255,
                        255,
                        255,
                    ]

    except Exception:
        pass

    return chessboard_image


def draw_axis(img, projected_corners, imgpts):
    """Draws the projected axis on the checkerboard"""

    try:
        pt1 = tuple(projected_corners[0].ravel())
        pt2x = tuple(imgpts[0].ravel())
        pt2y = tuple(imgpts[1].ravel())
        pt2z = tuple(imgpts[2].ravel())

        img = cv2.line(
            img, (int(pt1[0]), int(pt1[1])), (int(pt2x[0]), int(pt2x[1])), (255, 0, 0), 5
        )
        img = cv2.line(
            img, (int(pt1[0]), int(pt1[1])), (int(pt2y[0]), int(pt2y[1])), (0, 255, 0), 5
        )
        img = cv2.line(
            img, (int(pt1[0]), int(pt1[1])), (int(pt2z[0]), int(pt2z[1])), (0, 0, 255), 5
        )

    except Exception:
        pass

    return img


def draw_corners(img, corner_checkers: list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]]):
    """Draws the corners of the checkerboard on the image"""

    try:
        for checker, color in corner_checkers:
            points = np.array(checker, dtype=np.float32)
            cv2.fillPoly(img, [points.astype(np.int32)], color)  # type:ignore

    except Exception:
        pass

    return img


def get_checker_hists(
    frame: cv2.typing.MatLike,
    corner_checkers: list[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]],
    is_color_set: bool,
):
    """Calculates the histogram of the colored checkers and the non-colored checkers on the frame"""

    roi_hists: list[cv2.typing.MatLike] = []
    color_hists: list[tuple[cv2.typing.MatLike, cv2.typing.MatLike]] = []

    # hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    w, h = 100, 100  # or any dimensions you want
    b = 15

    color_range_lower = np.array((0.0, 60.0, 32.0))
    color_range_upper = np.array((180.0, 255.0, 255.0))

    corner_images = None
    for checker, _ in corner_checkers:
        # Define the vertices of your polygon (e.g., a quadrilateral)
        points = np.array(checker, dtype=np.float32)

        # Create a mask for the polygon
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)  # type:ignore
        # Invert the mask
        mask_inv = cv2.bitwise_not(mask)
        # Extract the checker polygon from the image using the mask
        polygon = cv2.bitwise_and(frame, frame, mask=mask)

        # Create the destination points for the perspective warp (edges of the checker)
        destination_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(points, destination_points)

        # Perform the perspective warp (backproject the polygon to a normal rectangle)
        transformed_polygon = cv2.warpPerspective(polygon, M, (w, h))
        # Crop it nicely
        final_crop = transformed_polygon[b : h - b, b : w - b]
        # Stack the checker ROIs
        corner_images = (
            final_crop if corner_images is None else cv2.vconcat([corner_images, final_crop])
        )

        # This is the histogram for MeanShift tracking (which we don't use)
        hsv_roi = cv2.cvtColor(final_crop, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv_roi, color_range_lower, color_range_upper)
        roi_hist = cv2.calcHist(
            images=[hsv_roi],
            channels=[0],
            mask=hsv_mask,
            histSize=[180],
            ranges=[0, 180],
        )
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        roi_hists.append(roi_hist)

        # We'll do this only once to create the Bayesian color model
        if not is_color_set:
            # Calculate the histogram of the ROI (colored checker) on the RGB image
            color_hist = cv2.calcHist(
                images=[frame],
                channels=[0, 1, 2],
                mask=mask,
                histSize=[32, 32, 32],
                ranges=[0, 256, 0, 256, 0, 256],
            )

            # Calculate the histogram of the non-ROI (everything else) on the RGB image
            non_color_hist = cv2.calcHist(
                images=[frame],
                channels=[0, 1, 2],
                mask=mask_inv,
                histSize=[32, 32, 32],
                ranges=[0, 256, 0, 256, 0, 256],
            )
            color_hists.append((color_hist, non_color_hist))

    if corner_images is not None:
        # Display the stacked checker ROIs
        cv2.namedWindow("corners", cv2.WINDOW_KEEPRATIO)
        cv2.imshow("corners", corner_images)

    return roi_hists, color_hists


async def get_camera_calibration(vid: int):
    """Get the camera calibration settings for the camera with the given video ID"""
    await asyncio.subprocess.create_subprocess_exec(
        "v4l2-ctl",
        *["--device", f"/dev/video{vid}", "-l"],
    )


async def set_camera_calibration(vid: int, dict: dict[str, str]):
    """Set the camera calibration settings for the camera with the given video ID"""
    settings = ["--device", f"/dev/video{vid}"]
    for k, v in dict.items():
        settings.append("-c")
        settings.append(f"{k}={v}")

    await asyncio.subprocess.create_subprocess_exec("v4l2-ctl", *settings)


async def lock_camera(vid: int = 0):
    """Lock the camera's Exposure and White Balance settings for the camera with the given video ID"""
    await set_camera_calibration(
        vid,
        {
            "auto_exposure": "1",  # 1 manual, 3 auto
            "exposure_dynamic_framerate": "0",  # 0 manual, 1 auto
            "white_balance_automatic": "0",  # 0 manual, 1 auto
        },
    )


async def main():
    print(cv2.getBuildInformation())
    await set_camera_calibration(
        0,
        {
            "auto_exposure": "3",  # 1 manual, 3 auto
            "exposure_time_absolute": "500",
            "exposure_dynamic_framerate": "1",  # 0 manual, 1 auto
            "white_balance_automatic": "1",  # 0 manual, 1 auto
            "white_balance_temperature": "6000",
        },
    )
    await get_camera_calibration(0)

    is_found = False
    is_calibrated = False
    is_found_past = False

    matrix_size = (CHESS_WIDTH, CHESS_HEIGHT)

    # Create a chessboard pattern in real world space (checker coordinates)
    objp = np.zeros((CHESS_WIDTH * CHESS_HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESS_WIDTH, 0:CHESS_HEIGHT].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in the image plane

    # The size of the Axis is three checkers long/wide/deep
    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    # term_crit_cam_shift = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # Create the corners of the colored checkers (in flat space)
    corner_dots = np.float32(
        [
            [0, 0, 0],  # red LT
            [0, -1, 0],
            [-1, -1, 0],
            [-1, 0, 0],
            [CHESS_WIDTH, 0, 0],  # purple RT
            [CHESS_WIDTH, -1, 0],
            [CHESS_WIDTH - 1, -1, 0],
            [CHESS_WIDTH - 1, 0, 0],
            [0, CHESS_HEIGHT, 0],  # orange LB
            [0, CHESS_HEIGHT - 1, 0],
            [-1, CHESS_HEIGHT - 1, 0],
            [-1, CHESS_HEIGHT, 0],
            [CHESS_WIDTH, CHESS_HEIGHT, 0],  # green RB
            [CHESS_WIDTH, CHESS_HEIGHT - 1, 0],
            [CHESS_WIDTH - 1, CHESS_HEIGHT - 1, 0],
            [CHESS_WIDTH - 1, CHESS_HEIGHT, 0],
        ]
    ).reshape(-1, 3)

    # The colors of the checkers
    corner_colors: list[tuple[int, ...]] = [
        (13, 158, 56),  # green
        (22, 140, 250),  # orange
        (133, 16, 57),  # purple
        (45, 34, 245),  # red
    ]

    vid = cv2.VideoCapture(0)
    cv2.namedWindow("window", cv2.WINDOW_KEEPRATIO)

    # track_windows: list[cv2.typing.Rect | None] = [None] * len(corner_colors)
    corner_points: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    nb_hist_fps = 0
    color_hists_sum: list[cv2.typing.MatLike | None] = [None] * 4
    non_color_hists_sum: list[cv2.typing.MatLike | None] = [None] * 4

    factor = None
    p_color_rgbs = None
    epsilon = np.finfo(float).eps

    while True:
        start_time = time()

        _, frame = vid.read()
        # Grayscale frame for the chessboard detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # We only draw on the canvas
        canvas = frame.copy()

        if not is_calibrated:
            is_found, corners = cv2.findChessboardCorners(gray_frame, matrix_size, None)

        if is_found and not is_calibrated:
            objpoints.append(objp)

            corners = cv2.cornerSubPix(
                gray_frame,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            # Add the real coordinates of the checker corners to the image plane
            imgpoints.append(corners)

            if len(objpoints) > NB_CALIB_CHECKS:
                _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objectPoints=objpoints,
                    imagePoints=imgpoints,
                    imageSize=matrix_size,
                    cameraMatrix=None,
                    distCoeffs=None,
                    flags=None,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.001,
                    ),
                )  # type: ignore

                objpoints = objpoints[-NB_CALIB_CHECKS:]
                imgpoints = imgpoints[-NB_CALIB_CHECKS:]
                is_calibrated = True

            cv2.drawChessboardCorners(canvas, matrix_size, corners, is_found)

        found2 = False
        if is_calibrated:
            ## We could undistort the image to make the images plance actually flat, but we won't do that

            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            #     mtx, dist, matrix_size, 1, matrix_size
            # )
            # frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            # canvas = frame.copy()

            found2, corners2 = cv2.findChessboardCorners(gray_frame, matrix_size, None)

            if found2:
                corners2 = cv2.cornerSubPix(
                    gray_frame,
                    corners2,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                # Use RANSAC to solve the mapping between the checkerboard and the real world
                ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
                is_found_past = True

            if is_found_past:
                corner_points = []
                # Project the corners of the colored checkers to the image plane
                # We do this efficiently by projecting all the corners of all colored chcekers at once
                proj_points, _ = cv2.projectPoints(corner_dots, rvecs, tvecs, mtx, dist)
                for pidx in range(0, len(proj_points) - 3, 4):
                    proj_pt1 = tuple(int(i) for i in proj_points[pidx].ravel())
                    proj_pt2 = tuple(int(i) for i in proj_points[pidx + 1].ravel())
                    proj_pt3 = tuple(int(i) for i in proj_points[pidx + 2].ravel())
                    proj_pt4 = tuple(int(i) for i in proj_points[pidx + 3].ravel())
                    corner_points.append(
                        ((proj_pt1, proj_pt2, proj_pt3, proj_pt4), corner_colors[pidx // 4])
                    )

            if corner_points:
                # if we found the checkerboard in this iteration, we can calculate all color histograms
                if found2:
                    _, color_hists = get_checker_hists(
                        frame, corner_points, p_color_rgbs is not None
                    )
                    # track_windows = [None] * len(corner_points)

                    # For each colored checker, sum the histograms
                    for cidx, (color_hist, non_color_hist) in enumerate(color_hists):
                        if color_hists_sum[cidx] is None:
                            color_hists_sum[cidx] = color_hist
                            non_color_hists_sum[cidx] = non_color_hist

                        else:
                            color_hists_sum[cidx] += color_hist
                            non_color_hists_sum[cidx] += non_color_hist

                    nb_hist_fps += 1

                # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # for cidx, (checker_hist, (corner_point, _)) in enumerate(
                #     zip(checker_hists, corner_points)
                # ):
                #     track_window = track_windows[cidx]
                #     if track_window is None:
                #         points = np.array(corner_point, dtype=np.int32).reshape(-1, 1, 2)
                #         track_window = cv2.boundingRect(points)

                #     dst = cv2.calcBackProject([hsv], [0], checker_hist, [0, 180], 1)
                #     _, track_window = cv2.meanShift(dst, track_window, term_crit_cam_shift)

                #     # ret, track_window = cv2.CamShift(dst, track_window, term_crit_cam_shift)
                #     # pts = cv2.boxPoints(ret)
                #     # pts = np.intp(pts)
                #     # canvas = cv2.polylines(canvas, [pts], True, (255, 255, 0), 2)  # type:ignore

                #     track_windows[cidx] = track_window

            # canvas = draw_corners(canvas, corner_points)

            # for track_window in track_windows:
            #     if track_window is not None:
            #         x, y, w, h = track_window
            #         cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # We reach the calibration threshold, we can calculate the color model
            if p_color_rgbs is None and nb_hist_fps > NB_CALIB_CHECKS:
                p_color_rgbs = []

                # FOR each colored checker
                for cidx in range(len(color_hists_sum)):
                    color_hist_sum = color_hists_sum[cidx]
                    assert color_hist_sum is not None
                    non_color_hist_sum = non_color_hists_sum[cidx]
                    assert non_color_hist_sum is not None
                    color_hist_sum_s = np.sum(
                        color_hist_sum
                    )  # Value for the total sum of the pixel count of each COLOR bin
                    non_color_hist_sum_n = np.sum(
                        non_color_hist_sum
                    )  # Value for the total sum of the pixel count of each NON-COLOR bin

                    # Hit it Bayes!
                    p_color = color_hist_sum_s / (color_hist_sum_s + non_color_hist_sum_n)
                    p_non_color = 1 - p_color

                    p_rgb_color = color_hist_sum / color_hist_sum_s
                    p_rgb_non_color = non_color_hist_sum / non_color_hist_sum_n

                    p_rgb = p_rgb_color * p_color + p_rgb_non_color * p_non_color
                    # Just to make sure we don't divide by zero
                    p_rgb[p_rgb == 0] = epsilon

                    # Add the color model to the list
                    p_color_rgbs.append(p_rgb_color * p_color / p_rgb)

                    # Calculate the scaling factor for the frame
                    factor = 1.0 / 256.0 * p_color_rgbs[-1].shape[0]

        # if the color models are calibrated
        if not (factor is None or p_color_rgbs is None):
            # Calculate indices
            indices = np.floor(frame * factor).astype(int)

            for pidx, p_color_rgb in enumerate(p_color_rgbs):
                # Fetch probabilities using advanced indexing
                probabilities = p_color_rgb[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
                # Paint the canvas at these locations
                canvas[probabilities > THETA] = corner_colors[pidx]

        if found2:
            # Draw the axis on the checkerboard if it was found using chessboard detection
            axis_points, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            canvas = draw_axis(canvas, corners2, axis_points)

        # Draw some values on the canvas
        fps = 1 / (time() - start_time)
        auto_exp = True if vid.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 3 else False
        cv2.putText(
            canvas,
            f"FPS: {fps:.2f}  EXP: {vid.get(cv2.CAP_PROP_EXPOSURE)}  WB: {vid.get(cv2.CAP_PROP_WB_TEMPERATURE)}  auto: {auto_exp}",
            (5, 18),
            cv2.FONT_ITALIC,
            0.6,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Calibrated: {is_calibrated}",
            (5, 38),
            cv2.FONT_ITALIC,
            0.6,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow("window", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            await lock_camera()

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
