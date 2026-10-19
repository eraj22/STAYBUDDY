import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';

class ParentAttendanceScreen extends StatefulWidget {
  const ParentAttendanceScreen({super.key});

  @override
  State<ParentAttendanceScreen> createState() =>
      _ParentAttendanceScreenState();
}

class _ParentAttendanceScreenState extends State<ParentAttendanceScreen> {
  late VideoPlayerController _videoController;

  static const Color background = Color(0xFF063E40);
  static const Color darkTeal = Color(0xFF074F51);
  static const Color teal = Color(0xFF0A6B6E);
  static const Color mint = Color(0xFF5DCAA5);
  static const Color green = Color(0xFF35EE96);
  static const Color red = Color(0xFFFF6B6B);
  static const Color warning = Color(0xFFFFD166);

  @override
  void initState() {
    super.initState();

    _videoController = VideoPlayerController.asset(
      'assets/images/background.mp4',
    )
      ..initialize().then((_) {
        if (!mounted) return;

        _videoController
          ..setLooping(true)
          ..setVolume(0)
          ..play();

        setState(() {});
      });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: background,
      body: Stack(
        children: [
          _buildBackground(),

          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(.46),
            ),
          ),

          SafeArea(
            child: Column(
              children: [
                _buildTopBar(),

                Expanded(
                  child: SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(
                      18,
                      10,
                      18,
                      30,
                    ),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(
                          maxWidth: 600,
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _buildCurrentStatusCard(),

                            const SizedBox(height: 20),

                            _sectionTitle(
                              'Today',
                              'Latest hostel entry and exit activity',
                            ),

                            const SizedBox(height: 12),

                            _buildTodayActivity(),

                            const SizedBox(height: 20),

                            _sectionTitle(
                              'Recent Attendance',
                              'Previous IN / OUT records',
                            ),

                            const SizedBox(height: 12),

                            _buildAttendanceHistory(),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBackground() {
    if (!_videoController.value.isInitialized) {
      return const Positioned.fill(
        child: ColoredBox(
          color: background,
        ),
      );
    }

    return Positioned.fill(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: _videoController.value.size.width,
          height: _videoController.value.size.height,
          child: VideoPlayer(_videoController),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        10,
        8,
        18,
        8,
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(
              Icons.arrow_back_ios_new_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),

          const SizedBox(width: 4),

          Expanded(
            child: Text(
              'Attendance',
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),

          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withOpacity(.07),
              border: Border.all(
                color: mint.withOpacity(.14),
              ),
            ),
            child: const Icon(
              Icons.fact_check_outlined,
              color: mint,
              size: 21,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCurrentStatusCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF00645E).withOpacity(.96),
            const Color(0xFF003B3B).withOpacity(.96),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: mint.withOpacity(.20),
        ),
        boxShadow: [
          BoxShadow(
            color: teal.withOpacity(.12),
            blurRadius: 25,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 66,
                height: 66,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withOpacity(.09),
                  border: Border.all(
                    color: mint.withOpacity(.7),
                  ),
                ),
                child: const Icon(
                  Icons.person_rounded,
                  color: Colors.white,
                  size: 40,
                ),
              ),

              const SizedBox(width: 14),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Ali Raza',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                      ),
                    ),

                    const SizedBox(height: 4),

                    Text(
                      'FAST National University',
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: mint,
                        fontSize: 11.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: green.withOpacity(.10),
              borderRadius: BorderRadius.circular(17),
              border: Border.all(
                color: green.withOpacity(.22),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: const BoxDecoration(
                    color: green,
                    shape: BoxShape.circle,
                  ),
                ),

                const SizedBox(width: 9),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Currently inside hostel',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 12.5,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        'Last entry: Today at 7:45 AM',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.45),
                          fontSize: 10.5,
                        ),
                      ),
                    ],
                  ),
                ),

                Text(
                  'IN',
                  style: GoogleFonts.dmSans(
                    color: green,
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(
    String title,
    String subtitle,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(
            color: Colors.white38,
            fontSize: 11,
          ),
        ),
      ],
    );
  }

  Widget _buildTodayActivity() {
    return _glassCard(
      child: Column(
        children: [
          _activityItem(
            icon: Icons.login_rounded,
            iconColor: green,
            title: 'Entered Hostel',
            time: '7:45 AM',
            subtitle: 'Main Gate',
            status: 'IN',
            statusColor: green,
          ),

          _divider(),

          _activityItem(
            icon: Icons.logout_rounded,
            iconColor: red,
            title: 'Left Hostel',
            time: '7:10 AM',
            subtitle: 'Main Gate',
            status: 'OUT',
            statusColor: red,
          ),

          _divider(),

          _activityItem(
            icon: Icons.login_rounded,
            iconColor: green,
            title: 'Entered Hostel',
            time: '9:20 PM',
            subtitle: 'Yesterday',
            status: 'IN',
            statusColor: green,
          ),
        ],
      ),
    );
  }

  Widget _buildAttendanceHistory() {
    return Column(
      children: [
        _attendanceDayCard(
          date: '25 Aug 2026',
          inTime: '7:40 AM',
          outTime: '6:15 PM',
          finalStatus: 'IN',
          finalColor: green,
        ),

        const SizedBox(height: 10),

        _attendanceDayCard(
          date: '24 Aug 2026',
          inTime: '8:10 AM',
          outTime: '5:55 PM',
          finalStatus: 'IN',
          finalColor: green,
        ),

        const SizedBox(height: 10),

        _attendanceDayCard(
          date: '23 Aug 2026',
          inTime: '8:00 AM',
          outTime: '10:15 PM',
          finalStatus: 'Late Entry',
          finalColor: warning,
        ),

        const SizedBox(height: 10),

        _attendanceDayCard(
          date: '22 Aug 2026',
          inTime: '7:50 AM',
          outTime: '6:00 PM',
          finalStatus: 'IN',
          finalColor: green,
        ),
      ],
    );
  }

  Widget _attendanceDayCard({
    required String date,
    required String inTime,
    required String outTime,
    required String finalStatus,
    required Color finalColor,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.32),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: mint.withOpacity(.10),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  date,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 5,
                ),
                decoration: BoxDecoration(
                  color: finalColor.withOpacity(.10),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  finalStatus,
                  style: GoogleFonts.dmSans(
                    color: finalColor,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 14),

          Row(
            children: [
              Expanded(
                child: _timeBox(
                  icon: Icons.login_rounded,
                  title: 'IN',
                  time: inTime,
                  color: green,
                ),
              ),

              const SizedBox(width: 10),

              Expanded(
                child: _timeBox(
                  icon: Icons.logout_rounded,
                  title: 'OUT',
                  time: outTime,
                  color: red,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _timeBox({
    required IconData icon,
    required String title,
    required String time,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 12,
        vertical: 11,
      ),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.10),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        children: [
          Icon(
            icon,
            color: color,
            size: 18,
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.dmSans(
                    color: color,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  time,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 11.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _activityItem({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String time,
    required String subtitle,
    required String status,
    required Color statusColor,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: 12,
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: iconColor.withOpacity(.10),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(
              icon,
              color: iconColor,
              size: 21,
            ),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 12.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  '$subtitle • $time',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white38,
                    fontSize: 10.5,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          Text(
            status,
            style: GoogleFonts.dmSans(
              color: statusColor,
              fontSize: 11,
              fontWeight: FontWeight.w800,
            ),
          ),
        ],
      ),
    );
  }

  Widget _glassCard({
    required Widget child,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(
        horizontal: 16,
        vertical: 4,
      ),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.32),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: mint.withOpacity(.10),
        ),
      ),
      child: child,
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      color: Colors.white.withOpacity(.06),
    );
  }
}
