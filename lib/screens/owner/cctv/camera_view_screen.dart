import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';

class CameraViewScreen extends StatefulWidget {
  final String hostelName;
  final Map<String, dynamic> camera;

  const CameraViewScreen({
    super.key,
    required this.hostelName,
    required this.camera,
  });

  @override
  State<CameraViewScreen> createState() => _CameraViewScreenState();
}

class _CameraViewScreenState extends State<CameraViewScreen> {
  bool _muted = true;
  bool _recording = true;
  bool _fullscreenMode = false;

  double _playbackPosition = 0.64;

  final List<String> _timeOptions = [
    'Live',
    '1 hr',
    '6 hrs',
    '24 hrs',
  ];

  String _selectedTime = 'Live';

  @override
  Widget build(BuildContext context) {
    final online = widget.camera['status'] == 'Online';

    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: CustomScrollView(
          physics: const BouncingScrollPhysics(),
          slivers: [
            SliverToBoxAdapter(
              child: _buildHeader(),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
              sliver: SliverToBoxAdapter(
                child: _buildLivePreview(
                  online: online,
                ),
              ),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
              sliver: SliverToBoxAdapter(
                child: _buildControls(),
              ),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
              sliver: SliverToBoxAdapter(
                child: _sectionTitle(
                  title: 'Playback',
                  subtitle: 'Review recent footage',
                ),
              ),
            ),
            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 18),
              sliver: SliverToBoxAdapter(
                child: _buildPlaybackCard(
                  online: online,
                ),
              ),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
              sliver: SliverToBoxAdapter(
                child: _sectionTitle(
                  title: 'Camera Information',
                  subtitle: 'Device and stream details',
                ),
              ),
            ),
            SliverPadding(
              padding: const EdgeInsets.symmetric(horizontal: 18),
              sliver: SliverToBoxAdapter(
                child: _buildCameraInfo(
                  online: online,
                ),
              ),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(18, 24, 18, 40),
              sliver: SliverToBoxAdapter(
                child: _buildSecurityNotice(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF073F42),
            Color(0xFF0A696D),
            Color(0xFF0A797B),
          ],
        ),
        borderRadius: BorderRadius.vertical(
          bottom: Radius.circular(32),
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(
            18,
            14,
            18,
            24,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: () => Navigator.pop(context),
                  ),
                  const Spacer(),
                  _circleButton(
                    icon: Icons.more_horiz_rounded,
                    onTap: _showMoreMenu,
                  ),
                ],
              ),
              const SizedBox(height: 22),
              Text(
                widget.camera['name']?.toString() ?? 'Camera',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.playfairDisplay(
                  fontSize: 28,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 5),
              Text(
                '${widget.camera['location']} · ${widget.hostelName}',
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 12,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _circleButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.white.withOpacity(0.18),
            ),
          ),
          child: Icon(
            icon,
            color: Colors.white,
            size: 20,
          ),
        ),
      ),
    );
  }

  Widget _buildLivePreview({
    required bool online,
  }) {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.14),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: AspectRatio(
        aspectRatio: 16 / 10,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(24),
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: online
                    ? [
                        const Color(0xFF17393B),
                        const Color(0xFF071D1E),
                      ]
                    : [
                        const Color(0xFF575B5C),
                        const Color(0xFF272A2B),
                      ],
              ),
            ),
            child: Stack(
              children: [
                Center(
                  child: Container(
                    width: 94,
                    height: 94,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.06),
                      shape: BoxShape.circle,
                    ),
                    child: Icon(
                      widget.camera['icon'] as IconData? ??
                          Icons.videocam_rounded,
                      size: 45,
                      color: Colors.white.withOpacity(0.65),
                    ),
                  ),
                ),

                Positioned(
                  top: 14,
                  left: 14,
                  child: Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 9,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: online
                              ? const Color(0xFFD9534F)
                              : const Color(0xFF696969),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          online ? 'LIVE' : 'OFFLINE',
                          style: GoogleFonts.dmSans(
                            fontSize: 8.5,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                      ),
                      const SizedBox(width: 7),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black.withOpacity(0.25),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          widget.camera['quality']?.toString() ?? 'HD',
                          style: GoogleFonts.dmSans(
                            fontSize: 8.5,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),

                if (_recording && online)
                  Positioned(
                    top: 14,
                    right: 14,
                    child: Row(
                      children: [
                        Container(
                          width: 8,
                          height: 8,
                          decoration: const BoxDecoration(
                            color: Color(0xFFD9534F),
                            shape: BoxShape.circle,
                          ),
                        ),
                        const SizedBox(width: 5),
                        Text(
                          'REC',
                          style: GoogleFonts.dmSans(
                            fontSize: 9,
                            fontWeight: FontWeight.w800,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),

                Positioned(
                  left: 14,
                  right: 14,
                  bottom: 14,
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          online
                              ? 'Secure live stream'
                              : 'Camera unavailable',
                          style: GoogleFonts.dmSans(
                            fontSize: 10.5,
                            color: Colors.white.withOpacity(0.68),
                          ),
                        ),
                      ),
                      Icon(
                        online
                            ? Icons.lock_rounded
                            : Icons.cloud_off_rounded,
                        size: 15,
                        color: Colors.white.withOpacity(0.68),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildControls() {
    return Row(
      children: [
        Expanded(
          child: _controlButton(
            icon: _muted
                ? Icons.volume_off_rounded
                : Icons.volume_up_rounded,
            label: _muted ? 'Muted' : 'Sound',
            active: !_muted,
            onTap: () {
              setState(() {
                _muted = !_muted;
              });
            },
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _controlButton(
            icon: Icons.photo_camera_outlined,
            label: 'Snapshot',
            onTap: () {
              _showTemporary(
                'Snapshot capture will be connected with live CCTV integration later.',
              );
            },
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _controlButton(
            icon: _fullscreenMode
                ? Icons.fullscreen_exit_rounded
                : Icons.fullscreen_rounded,
            label: 'Fullscreen',
            active: _fullscreenMode,
            onTap: () {
              setState(() {
                _fullscreenMode = !_fullscreenMode;
              });

              _showTemporary(
                _fullscreenMode
                    ? 'Fullscreen preview enabled in UI.'
                    : 'Fullscreen preview closed.',
              );
            },
          ),
        ),
      ],
    );
  }

  Widget _controlButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool active = false,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(17),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          padding: const EdgeInsets.symmetric(
            horizontal: 8,
            vertical: 13,
          ),
          decoration: BoxDecoration(
            color: active
                ? AppTheme.teal.withOpacity(0.09)
                : AppTheme.card,
            borderRadius: BorderRadius.circular(17),
            border: Border.all(
              color: active
                  ? AppTheme.teal.withOpacity(0.45)
                  : AppTheme.border,
            ),
          ),
          child: Column(
            children: [
              Icon(
                icon,
                size: 20,
                color: active
                    ? AppTheme.teal
                    : AppTheme.textMuted,
              ),
              const SizedBox(height: 5),
              Text(
                label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 9.5,
                  fontWeight: FontWeight.w600,
                  color: active
                      ? AppTheme.teal
                      : AppTheme.textPrimary,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionTitle({
    required String title,
    required String subtitle,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.playfairDisplay(
            fontSize: 19,
            fontWeight: FontWeight.w700,
            color: AppTheme.textPrimary,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(
            fontSize: 10.5,
            color: AppTheme.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _buildPlaybackCard({
    required bool online,
  }) {
    return Container(
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            height: 38,
            child: ListView.separated(
              scrollDirection: Axis.horizontal,
              itemCount: _timeOptions.length,
              separatorBuilder: (_, __) =>
                  const SizedBox(width: 8),
              itemBuilder: (context, index) {
                final option = _timeOptions[index];
                final selected = option == _selectedTime;

                return GestureDetector(
                  onTap: () {
                    setState(() {
                      _selectedTime = option;
                    });
                  },
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 180),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 13,
                      vertical: 8,
                    ),
                    decoration: BoxDecoration(
                      color: selected
                          ? AppTheme.teal
                          : AppTheme.bgSecondary,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      option,
                      style: GoogleFonts.dmSans(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w600,
                        color: selected
                            ? Colors.white
                            : AppTheme.textMuted,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),

          const SizedBox(height: 22),

          Row(
            children: [
              Text(
                _selectedTime == 'Live'
                    ? 'LIVE'
                    : 'Recorded footage',
                style: GoogleFonts.dmSans(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: _selectedTime == 'Live'
                      ? const Color(0xFFD9534F)
                      : AppTheme.textPrimary,
                ),
              ),
              const Spacer(),
              Text(
                _selectedTime == 'Live'
                    ? 'Now'
                    : '08:40 PM',
                style: GoogleFonts.dmSans(
                  fontSize: 10,
                  color: AppTheme.textMuted,
                ),
              ),
            ],
          ),

          const SizedBox(height: 10),

          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              trackHeight: 5,
              thumbShape: const RoundSliderThumbShape(
                enabledThumbRadius: 7,
              ),
              overlayShape: const RoundSliderOverlayShape(
                overlayRadius: 14,
              ),
            ),
            child: Slider(
              value: _selectedTime == 'Live'
                  ? 1
                  : _playbackPosition,
              activeColor: AppTheme.teal,
              inactiveColor: AppTheme.teal.withOpacity(0.10),
              onChanged: _selectedTime == 'Live'
                  ? null
                  : (value) {
                      setState(() {
                        _playbackPosition = value;
                      });
                    },
            ),
          ),

          const SizedBox(height: 4),

          Row(
            children: [
              Text(
                _selectedTime == 'Live' ? 'Live stream' : '08:00 PM',
                style: GoogleFonts.dmSans(
                  fontSize: 9,
                  color: AppTheme.textMuted,
                ),
              ),
              const Spacer(),
              Text(
                _selectedTime == 'Live' ? 'Now' : '09:00 PM',
                style: GoogleFonts.dmSans(
                  fontSize: 9,
                  color: AppTheme.textMuted,
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _playbackButton(
                Icons.replay_10_rounded,
                () {
                  _showTemporary('Moved 10 seconds backward in UI.');
                },
              ),
              const SizedBox(width: 16),
              _playbackPrimaryButton(
                online
                    ? Icons.pause_rounded
                    : Icons.play_arrow_rounded,
                () {
                  _showTemporary(
                    'Playback controls will connect to actual CCTV footage later.',
                  );
                },
              ),
              const SizedBox(width: 16),
              _playbackButton(
                Icons.forward_10_rounded,
                () {
                  _showTemporary('Moved 10 seconds forward in UI.');
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _playbackButton(
    IconData icon,
    VoidCallback onTap,
  ) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            color: AppTheme.bgSecondary,
            shape: BoxShape.circle,
          ),
          child: Icon(
            icon,
            color: AppTheme.textPrimary,
            size: 21,
          ),
        ),
      ),
    );
  }

  Widget _playbackPrimaryButton(
    IconData icon,
    VoidCallback onTap,
  ) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 54,
          height: 54,
          decoration: const BoxDecoration(
            color: AppTheme.teal,
            shape: BoxShape.circle,
          ),
          child: Icon(
            icon,
            color: Colors.white,
            size: 26,
          ),
        ),
      ),
    );
  }

  Widget _buildCameraInfo({
    required bool online,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 16,
        vertical: 4,
      ),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Column(
        children: [
          _infoRow(
            icon: Icons.videocam_outlined,
            label: 'Camera Name',
            value: widget.camera['name']?.toString() ?? '-',
          ),
          _divider(),
          _infoRow(
            icon: Icons.location_on_outlined,
            label: 'Location',
            value: widget.camera['location']?.toString() ?? '-',
          ),
          _divider(),
          _infoRow(
            icon: Icons.wifi_rounded,
            label: 'Connection',
            value: online ? 'Online' : 'Offline',
          ),
          _divider(),
          _infoRow(
            icon: Icons.high_quality_outlined,
            label: 'Stream Quality',
            value: widget.camera['quality']?.toString() ?? 'HD',
          ),
          _divider(),
          _infoRow(
            icon: Icons.fiber_manual_record_rounded,
            label: 'Recording',
            value: _recording ? 'Enabled' : 'Disabled',
          ),
          _divider(),
          _infoRow(
            icon: Icons.update_rounded,
            label: 'Last Updated',
            value:
                widget.camera['lastUpdated']?.toString() ?? 'Unknown',
          ),
        ],
      ),
    );
  }

  Widget _infoRow({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: 13,
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.07),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              size: 18,
              color: AppTheme.teal,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    color: AppTheme.textMuted,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      thickness: 0.6,
      color: AppTheme.border,
    );
  }

  Widget _buildSecurityNotice() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF4E6),
        borderRadius: BorderRadius.circular(19),
        border: Border.all(
          color: const Color(0xFFC07421).withOpacity(0.18),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.shield_outlined,
            size: 21,
            color: Color(0xFFC07421),
          ),
          const SizedBox(width: 11),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Privacy & Security',
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w700,
                    color: const Color(0xFFC07421),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'CCTV footage should only be accessible to authorized hostel staff. '
                  'Avoid cameras in bedrooms, bathrooms, or other private areas.',
                  style: GoogleFonts.dmSans(
                    fontSize: 10,
                    height: 1.45,
                    color: const Color(0xFFC07421),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showMoreMenu() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: AppTheme.bg,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _menuTile(
                  icon: Icons.settings_input_component_outlined,
                  title: 'Camera Settings',
                  onTap: () {
                    Navigator.pop(context);
                    _showTemporary(
                      'Camera configuration will be connected later.',
                    );
                  },
                ),
                _menuTile(
                  icon: Icons.download_outlined,
                  title: 'Download Recording',
                  onTap: () {
                    Navigator.pop(context);
                    _showTemporary(
                      'Recording download will be connected later.',
                    );
                  },
                ),
                _menuTile(
                  icon: _recording
                      ? Icons.stop_circle_outlined
                      : Icons.fiber_manual_record_outlined,
                  title: _recording
                      ? 'Stop Recording'
                      : 'Start Recording',
                  onTap: () {
                    Navigator.pop(context);

                    setState(() {
                      _recording = !_recording;
                    });

                    _showTemporary(
                      _recording
                          ? 'Recording enabled in UI.'
                          : 'Recording disabled in UI.',
                    );
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _menuTile({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(14),
      ),
      leading: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.07),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(
          icon,
          color: AppTheme.teal,
          size: 19,
        ),
      ),
      title: Text(
        title,
        style: GoogleFonts.dmSans(
          fontSize: 12.5,
          fontWeight: FontWeight.w600,
          color: AppTheme.textPrimary,
        ),
      ),
      trailing: const Icon(
        Icons.chevron_right_rounded,
      ),
    );
  }

  void _showTemporary(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Text(
            message,
            style: GoogleFonts.dmSans(
              color: Colors.white,
            ),
          ),
        ),
      );
  }
}